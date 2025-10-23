import gc
import logging

import torch
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Advanced memory monitoring utility with peak tracking and OOM prediction."""

    def __init__(self, logger=None, detailed=False, interval=500, warning_threshold=0.75, danger_threshold=0.85):
        self.logger = logger or logging.getLogger(__name__)
        self.warning_threshold = warning_threshold
        self.danger_threshold = danger_threshold
        self.log_interval = interval
        self.detailed = detailed
        self.peak_allocated = 0.0
        self.peak_reserved = 0.0
        self.memory_history = []
        self.oom_warnings = 0

    def log_memory(self, stage: str, step: int, predict_oom: bool = True):
        """Log GPU memory usage with peak tracking and OOM prediction."""
        if not self.detailed:
            return
        elif not step % self.log_interval == 0:
            return
        elif not torch.cuda.is_available():
            return

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated

        # Track peaks
        self.peak_allocated = max(self.peak_allocated, allocated)
        self.peak_reserved = max(self.peak_reserved, reserved)

        # Store history for trend analysis
        self.memory_history.append({
            'stage': stage,
            'allocated': allocated,
            'reserved': reserved,
            'free': free,
            'usage_percent': allocated/total
        })

        # Keep only last N entries for trend analysis
        if len(self.memory_history) > 1000:
            self.memory_history.pop(0)

        status = ""
        usage_pct = allocated/total*100
        if usage_pct > 85:
            status = "⚠️  HIGH"
        elif usage_pct > 75:
            status = "📊 MODERATE"
        else:
            status = "✅ NORMAL"

        level = logging.INFO if usage_pct <= 85 else logging.WARNING
        logger.log(level, f"training_step   ={step:6d} | {stage:24s} | allocated={allocated:4.1f} / {total:4.1f} [{usage_pct:3.0f}%] | "
                   f"reserved={reserved:4.1f} | free={free:4.1f} | {status} | peak={self.peak_allocated:5.1f}")

        # OOM prediction and warnings
        if predict_oom:
            self._check_oom_risk(stage, allocated, total)

    def _check_oom_risk(self, stage: str, allocated: float, total: float):
        """Check for OOM risk and provide warnings."""
        usage_percent = allocated / total

        if usage_percent >= self.danger_threshold:
            self.oom_warnings += 1
            self.logger.error(f"[{stage}] CRITICAL MEMORY USAGE: {usage_percent*100:.1f}% - IMMINENT OOM RISK!")
            self.logger.error(f"  Only {(total-allocated)*1024:.1f}MB remaining")

            # Provide trend analysis if we have history
            if len(self.memory_history) >= 5:
                recent_growth = self._calculate_memory_trend()
                if recent_growth > 0.01:  # Growing by >1% per step
                    self.logger.error(f"  Memory growing rapidly: +{recent_growth*100:.2f}% per step")
                    self.logger.error(f"  Estimated steps until OOM: {max(1, int((1.0-usage_percent)/recent_growth))}")

        elif usage_percent >= self.warning_threshold:
            self.logger.warning(f"[{stage}] HIGH MEMORY USAGE: {usage_percent*100:.1f}% - Monitor closely")

            # Suggest optimizations
            if len(self.memory_history) >= 3:
                recent_growth = self._calculate_memory_trend()
                if recent_growth > 0.005:  # Growing by >0.5% per step
                    self.logger.warning(f"  Memory trending upward: +{recent_growth*100:.2f}% per step")
                    self.logger.warning(f"  Consider: torch.cuda.empty_cache(), reduce batch size, or gradient checkpointing")

    def _calculate_memory_trend(self):
        """Calculate recent memory usage trend."""
        if len(self.memory_history) < 3:
            return 0.0

        recent = self.memory_history[-5:]  # Last 5 measurements
        if len(recent) < 2:
            return 0.0

        # Simple linear trend
        return (recent[-1]['usage_percent'] - recent[0]['usage_percent']) / len(recent)

    def get_memory_summary(self):
        """Get comprehensive memory usage summary."""
        if not torch.cuda.is_available():
            return {}

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3

        return {
            'current_allocated_gb': allocated,
            'current_reserved_gb': reserved,
            'peak_allocated_gb': self.peak_allocated,
            'peak_reserved_gb': self.peak_reserved,
            'total_gb': total,
            'current_usage_percent': allocated/total*100,
            'peak_usage_percent': self.peak_allocated/total*100,
            'oom_warnings_count': self.oom_warnings,
            'memory_trend': self._calculate_memory_trend() if len(self.memory_history) >= 3 else 0.0
        }

    def log_and_clear_memory(self, stage, step):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        gc.collect()
        self.log_memory(stage, step)
