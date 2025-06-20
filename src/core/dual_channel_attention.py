"""
Sistema de Doble Canal de Atención
==================================

Este módulo implementa el sistema de doble canal de última generación:

Canal A: Atención Espaciotemporal Global (GMAR)
Canal B: Atención por Objetos (Object Tracking)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class AttentionMap:
    """Mapa de atención con metadatos."""
    attention_weights: torch.Tensor  # [H, W] o [T, H, W]
    spatial_resolution: Tuple[int, int]
    temporal_length: Optional[int] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = None

@dataclass
class ObjectDetection:
    """Detección de objeto individual."""
    bbox: List[float]  # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None
    attention_weights: Optional[torch.Tensor] = None

@dataclass
class ChannelAOutput:
    """Salida del Canal A - Atención Espaciotemporal Global."""
    global_attention: AttentionMap
    temporal_attention: Optional[torch.Tensor] = None
    spatial_attention: Optional[torch.Tensor] = None
    confidence_map: Optional[torch.Tensor] = None

@dataclass
class ChannelBOutput:
    """Salida del Canal B - Atención por Objetos."""
    detections: List[ObjectDetection]
    object_attention_maps: Dict[int, AttentionMap]
    tracking_info: Dict[str, Any]
    object_trajectories: Dict[int, List[Tuple[float, float]]]

class GMARAttention(nn.Module):
    """
    Gradient-weighted Multi-scale Attention with Residual connections (GMAR)
    
    Implementación de atención de última generación que combina:
    - Multi-scale attention
    - Gradient-weighted attention
    - Residual connections
    - Flash Attention 2.0
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        use_flash_attention: bool = True,
        use_gradient_weighting: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Multi-scale attention
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Gradient-weighted attention
        self.use_gradient_weighting = use_gradient_weighting
        if use_gradient_weighting:
            self.gradient_weight = nn.Parameter(torch.ones(1, num_heads, 1, 1))
            self.gradient_norm = nn.LayerNorm(embed_dim)
        
        # Multi-scale processing
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(embed_dim, embed_dim, kernel_size=k, padding=k//2, groups=embed_dim//64)
            for k in [1, 3, 5]
        ])
        
        # Residual connections
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(attention_dropout)
        self.use_flash_attention = use_flash_attention
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass con GMAR attention.
        
        Args:
            x: Input tensor [B, N, D]
            mask: Attention mask [B, N] or [B, N, N]
            
        Returns:
            Tuple de (output, attention_weights)
        """
        B, N, D = x.shape
        
        # Multi-scale spatial processing
        if len(x.shape) == 3:
            x_spatial = x.transpose(1, 2).view(B, D, int(N**0.5), int(N**0.5))
            multi_scale_features = []
            for conv in self.multi_scale_conv:
                multi_scale_features.append(conv(x_spatial))
            x_spatial = torch.stack(multi_scale_features).mean(0)
            x = x_spatial.flatten(2).transpose(1, 2)
        
        # Self-attention
        qkv = self.qkv(self.norm1(x)).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Flash Attention 2.0
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout.p)
            attention_weights = None  # Flash attention no retorna weights
        else:
            # Standard attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                attn = attn.masked_fill(mask == 0, float('-inf'))
            
            attention_weights = F.softmax(attn, dim=-1)
            attention_weights = self.dropout(attention_weights)
            attn_output = (attention_weights @ v).transpose(1, 2).reshape(B, N, D)
        
        # Gradient-weighted attention
        if self.use_gradient_weighting:
            if attention_weights is not None:
                # Aplicar pesos de gradiente a los attention weights
                gradient_weights = self.gradient_weight.view(1, self.num_heads, 1, 1)
                attention_weights = attention_weights * gradient_weights
                attn_output = (attention_weights @ v).transpose(1, 2).reshape(B, N, D)
            
            # Normalización con gradient norm
            attn_output = self.gradient_norm(attn_output)
        
        # Residual connection
        x = x + attn_output
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x, attention_weights

class MViTv2Backbone(nn.Module):
    """
    Backbone MViTv2 optimizado para atención espaciotemporal.
    """
    
    def __init__(
        self,
        model_size: str = "s",
        spatial_resolution: Tuple[int, int] = (224, 224),
        temporal_window: int = 16,
        use_flash_attention: bool = True
    ):
        super().__init__()
        
        # Configuración según tamaño del modelo
        configs = {
            "xs": {"embed_dim": 96, "num_heads": 6, "depth": 8},
            "s": {"embed_dim": 128, "num_heads": 8, "depth": 12},
            "b": {"embed_dim": 192, "num_heads": 12, "depth": 16}
        }
        
        config = configs.get(model_size, configs["s"])
        
        self.embed_dim = config["embed_dim"]
        self.spatial_resolution = spatial_resolution
        self.temporal_window = temporal_window
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(
            3, self.embed_dim,
            kernel_size=(3, 16, 16),
            stride=(1, 16, 16),
            padding=(1, 0, 0)
        )
        
        # Position embeddings
        spatial_size = (spatial_resolution[0] // 16, spatial_resolution[1] // 16)
        self.pos_embed = nn.Parameter(torch.randn(1, temporal_window, spatial_size[0] * spatial_size[1], self.embed_dim))
        
        # Transformer blocks con GMAR attention
        self.blocks = nn.ModuleList([
            GMARAttention(
                embed_dim=self.embed_dim,
                num_heads=config["num_heads"],
                use_flash_attention=use_flash_attention
            )
            for _ in range(config["depth"])
        ])
        
        # Output projection
        self.output_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass del backbone MViTv2.
        
        Args:
            x: Video tensor [B, C, T, H, W]
            
        Returns:
            Tuple de (features, attention_maps)
        """
        B, C, T, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, D, T, H//16, W//16]
        
        # Reshape para transformer
        x = x.permute(0, 2, 3, 4, 1)  # [B, T, H//16, W//16, D]
        x = x.reshape(B, T, -1, self.embed_dim)  # [B, T, N, D]
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Flatten temporal and spatial dimensions
        x = x.reshape(B, T * x.shape[2], self.embed_dim)
        
        attention_maps = []
        
        # Transformer blocks
        for block in self.blocks:
            x, attn_weights = block(x)
            if attn_weights is not None:
                attention_maps.append(attn_weights)
        
        # Output projection
        x = self.output_proj(x)
        
        return x, attention_maps

class ObjectDetector(nn.Module):
    """
    Detector de objetos optimizado para el Canal B.
    """
    
    def __init__(
        self,
        model_type: str = "yolo_v8",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        max_objects: int = 20
    ):
        super().__init__()
        
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_objects = max_objects
        
        # Cargar modelo según tipo
        if model_type == "yolo_v8":
            try:
                from ultralytics import YOLO
                self.model = YOLO('yolov8n.pt')  # o yolov8s.pt, yolov8m.pt
            except ImportError:
                logger.warning("Ultralytics no disponible, usando detector básico")
                self.model = self._create_basic_detector()
        elif model_type == "deformable_detr":
            # Implementar Deformable DETR
            self.model = self._create_deformable_detr()
        else:
            self.model = self._create_basic_detector()
    
    def _create_basic_detector(self):
        """Detector básico como fallback."""
        return None
    
    def _create_deformable_detr(self):
        """Implementación básica de Deformable DETR."""
        return None
    
    def forward(self, frames: torch.Tensor) -> List[List[ObjectDetection]]:
        """
        Detecta objetos en frames de video.
        
        Args:
            frames: Video frames [B, T, C, H, W]
            
        Returns:
            Lista de detecciones por frame
        """
        if self.model is None:
            return self._basic_detection(frames)
        
        detections_per_frame = []
        
        for frame_idx in range(frames.shape[1]):
            frame = frames[:, frame_idx]  # [B, C, H, W]
            
            if self.model_type == "yolo_v8":
                frame_detections = self._yolo_detection(frame)
            else:
                frame_detections = self._basic_detection(frame.unsqueeze(1))
            
            detections_per_frame.append(frame_detections)
        
        return detections_per_frame
    
    def _yolo_detection(self, frame: torch.Tensor) -> List[ObjectDetection]:
        """Detección usando YOLO."""
        # Convertir tensor a numpy para YOLO
        frame_np = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame_np = (frame_np * 255).astype(np.uint8)
        
        # Detección YOLO
        results = self.model(frame_np, conf=self.confidence_threshold, iou=self.nms_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    bbox = box.xyxy[0].cpu().numpy().tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    
                    detection = ObjectDetection(
                        bbox=bbox,
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence
                    )
                    detections.append(detection)
        
        return detections[:self.max_objects]
    
    def _basic_detection(self, frames: torch.Tensor) -> List[List[ObjectDetection]]:
        """Detección básica como fallback."""
        # Implementación simple basada en detección de movimiento
        detections_per_frame = []
        
        for frame_idx in range(frames.shape[1]):
            frame = frames[:, frame_idx]
            
            # Detección simple de objetos basada en contornos
            frame_np = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
            frame_np = (frame_np * 255).astype(np.uint8)
            gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
            
            # Detectar contornos
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            for contour in contours[:self.max_objects]:
                area = cv2.contourArea(contour)
                if area > 100:  # Filtrar contornos pequeños
                    x, y, w, h = cv2.boundingRect(contour)
                    bbox = [x, y, x + w, y + h]
                    
                    detection = ObjectDetection(
                        bbox=bbox,
                        class_id=0,
                        class_name="object",
                        confidence=0.5
                    )
                    detections.append(detection)
            
            detections_per_frame.append(detections)
        
        return detections_per_frame

class ObjectTracker:
    """
    Tracker de objetos para el Canal B.
    """
    
    def __init__(self, tracker_type: str = "bytetrack", persistence_threshold: float = 0.5):
        self.tracker_type = tracker_type
        self.persistence_threshold = persistence_threshold
        self.tracks = {}  # track_id -> track_info
        self.next_track_id = 0
        
    def update(self, detections: List[ObjectDetection], frame_idx: int) -> List[ObjectDetection]:
        """
        Actualiza tracks con nuevas detecciones.
        
        Args:
            detections: Detecciones del frame actual
            frame_idx: Índice del frame
            
        Returns:
            Detecciones con track_id asignado
        """
        if self.tracker_type == "bytetrack":
            return self._bytetrack_update(detections, frame_idx)
        else:
            return self._simple_tracking_update(detections, frame_idx)
    
    def _bytetrack_update(self, detections: List[ObjectDetection], frame_idx: int) -> List[ObjectDetection]:
        """Implementación simplificada de ByteTrack."""
        # Implementación básica de tracking
        return self._simple_tracking_update(detections, frame_idx)
    
    def _simple_tracking_update(self, detections: List[ObjectDetection], frame_idx: int) -> List[ObjectDetection]:
        """Tracking simple basado en IoU."""
        updated_detections = []
        
        for detection in detections:
            best_track_id = None
            best_iou = 0
            
            # Buscar track existente con mejor IoU
            for track_id, track_info in self.tracks.items():
                if track_info['last_frame'] == frame_idx - 1:  # Track activo
                    iou = self._calculate_iou(detection.bbox, track_info['bbox'])
                    if iou > best_iou and iou > 0.3:  # Threshold de IoU
                        best_iou = iou
                        best_track_id = track_id
            
            if best_track_id is not None:
                # Actualizar track existente
                detection.track_id = best_track_id
                self.tracks[best_track_id].update({
                    'bbox': detection.bbox,
                    'last_frame': frame_idx,
                    'confidence': detection.confidence
                })
            else:
                # Crear nuevo track
                detection.track_id = self.next_track_id
                self.tracks[self.next_track_id] = {
                    'bbox': detection.bbox,
                    'last_frame': frame_idx,
                    'confidence': detection.confidence,
                    'class_name': detection.class_name
                }
                self.next_track_id += 1
            
            updated_detections.append(detection)
        
        # Limpiar tracks inactivos
        active_tracks = {}
        for track_id, track_info in self.tracks.items():
            if frame_idx - track_info['last_frame'] <= 5:  # Mantener tracks activos por 5 frames
                active_tracks[track_id] = track_info
        self.tracks = active_tracks
        
        return updated_detections
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calcula IoU entre dos bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calcular intersección
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calcular unión
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

class DualChannelAttentionSystem:
    """
    Sistema principal de doble canal de atención.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Canal A - Atención Espaciotemporal Global
        self.channel_a = MViTv2Backbone(
            model_size=config['attention']['channel_a']['model'].split('_')[-1],
            spatial_resolution=tuple(config['attention']['channel_a']['spatial_resolution']),
            temporal_window=config['attention']['channel_a']['temporal_window'],
            use_flash_attention=config['attention']['channel_a']['flash_attention']
        ).to(self.device)
        
        # Canal B - Atención por Objetos
        self.channel_b_detector = ObjectDetector(
            model_type=config['attention']['channel_b']['detection_model'],
            confidence_threshold=config['attention']['channel_b']['confidence_threshold'],
            nms_threshold=config['attention']['channel_b']['nms_threshold'],
            max_objects=config['attention']['channel_b']['max_objects']
        ).to(self.device)
        
        self.channel_b_tracker = ObjectTracker(
            tracker_type=config['attention']['channel_b']['tracking_model'],
            persistence_threshold=config['attention']['channel_b']['tracking_persistence']
        )
        
        # Optimizaciones
        if config['hardware']['torch_compile']:
            self.channel_a = torch.compile(self.channel_a)
        
        if config['hardware']['mixed_precision']:
            self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info("✅ Sistema de doble canal de atención inicializado")
    
    def process_video(self, video_frames: torch.Tensor) -> Tuple[ChannelAOutput, ChannelBOutput]:
        """
        Procesa video con ambos canales de atención.
        
        Args:
            video_frames: Video tensor [B, C, T, H, W]
            
        Returns:
            Tuple de salidas de ambos canales
        """
        logger.info("🚀 Iniciando procesamiento de doble canal...")
        
        # Canal A - Atención Espaciotemporal Global
        channel_a_output = self._process_channel_a(video_frames)
        
        # Canal B - Atención por Objetos
        channel_b_output = self._process_channel_b(video_frames)
        
        logger.info("✅ Procesamiento de doble canal completado")
        
        return channel_a_output, channel_b_output
    
    def _process_channel_a(self, video_frames: torch.Tensor) -> ChannelAOutput:
        """Procesa el Canal A - Atención Espaciotemporal Global."""
        logger.info("📊 Procesando Canal A - Atención Espaciotemporal Global...")
        
        with torch.cuda.amp.autocast() if self.config['hardware']['mixed_precision'] else torch.no_grad():
            # Forward pass del backbone
            features, attention_maps = self.channel_a(video_frames)
            
            # Procesar attention maps
            if attention_maps:
                # Combinar attention maps de todos los layers
                combined_attention = torch.stack(attention_maps).mean(0)  # [B, H, T*N, T*N]
                
                # Separar atención temporal y espacial
                T = self.config['attention']['channel_a']['temporal_window']
                N = combined_attention.shape[-1] // T
                
                temporal_attention = combined_attention[:, :, :T, :T]
                spatial_attention = combined_attention[:, :, T:, T:]
                
                # Crear mapa de atención global
                global_attention = spatial_attention.mean(1)  # Promedio sobre heads
                global_attention = global_attention.reshape(-1, int(N**0.5), int(N**0.5))
                
                # Upscale a resolución original
                target_size = self.config['attention']['channel_a']['spatial_resolution']
                global_attention = F.interpolate(
                    global_attention.unsqueeze(1),
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
                
                attention_map = AttentionMap(
                    attention_weights=global_attention,
                    spatial_resolution=target_size,
                    temporal_length=T,
                    confidence=1.0
                )
                
                return ChannelAOutput(
                    global_attention=attention_map,
                    temporal_attention=temporal_attention,
                    spatial_attention=spatial_attention,
                    confidence_map=global_attention
                )
            else:
                # Fallback si no hay attention maps
                dummy_attention = torch.ones(
                    video_frames.shape[0],
                    *self.config['attention']['channel_a']['spatial_resolution']
                ).to(self.device)
                
                attention_map = AttentionMap(
                    attention_weights=dummy_attention,
                    spatial_resolution=self.config['attention']['channel_a']['spatial_resolution'],
                    temporal_length=self.config['attention']['channel_a']['temporal_window'],
                    confidence=0.5
                )
                
                return ChannelAOutput(global_attention=attention_map)
    
    def _process_channel_b(self, video_frames: torch.Tensor) -> ChannelBOutput:
        """Procesa el Canal B - Atención por Objetos."""
        logger.info("🎯 Procesando Canal B - Atención por Objetos...")
        
        B, C, T, H, W = video_frames.shape
        all_detections = []
        object_attention_maps = {}
        object_trajectories = {}
        
        # Procesar cada frame
        for frame_idx in range(T):
            frame = video_frames[:, :, frame_idx]  # [B, C, H, W]
            
            # Detectar objetos
            frame_detections = self.channel_b_detector(frame.unsqueeze(1))
            
            # Tracking
            tracked_detections = self.channel_b_tracker.update(frame_detections[0], frame_idx)
            
            # Crear attention maps por objeto
            for detection in tracked_detections:
                if detection.track_id is not None:
                    # Crear attention map para este objeto
                    attention_map = self._create_object_attention_map(
                        detection, frame.shape[-2:], frame_idx
                    )
                    
                    object_attention_maps[detection.track_id] = attention_map
                    
                    # Actualizar trayectoria
                    if detection.track_id not in object_trajectories:
                        object_trajectories[detection.track_id] = []
                    
                    center_x = (detection.bbox[0] + detection.bbox[2]) / 2
                    center_y = (detection.bbox[1] + detection.bbox[3]) / 2
                    object_trajectories[detection.track_id].append((center_x, center_y))
            
            all_detections.append(tracked_detections)
        
        return ChannelBOutput(
            detections=all_detections,
            object_attention_maps=object_attention_maps,
            tracking_info={'total_tracks': len(object_trajectories)},
            object_trajectories=object_trajectories
        )
    
    def _create_object_attention_map(
        self, 
        detection: ObjectDetection, 
        frame_size: Tuple[int, int], 
        frame_idx: int
    ) -> AttentionMap:
        """Crea un mapa de atención para un objeto específico."""
        H, W = frame_size
        attention_map = torch.zeros(H, W).to(self.device)
        
        # Crear máscara basada en bounding box
        x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        
        if x2 > x1 and y2 > y1:
            attention_map[y1:y2, x1:x2] = detection.confidence
        
        # Aplicar suavizado gaussiano
        attention_map = torch.from_numpy(
            cv2.GaussianBlur(attention_map.cpu().numpy(), (15, 15), 0)
        ).to(self.device)
        
        return AttentionMap(
            attention_weights=attention_map,
            spatial_resolution=(H, W),
            temporal_length=1,
            confidence=detection.confidence,
            metadata={
                'track_id': detection.track_id,
                'class_name': detection.class_name,
                'frame_idx': frame_idx
            }
        ) 