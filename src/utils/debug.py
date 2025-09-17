"""
Unified debugging system for SeedVR2 generation pipeline

Provides structured logging, memory tracking, and performance monitoring
for all pipeline stages, including BlockSwap operations.
"""

import time
import torch
import gc
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from src.optimization.memory_manager import get_vram_usage, get_basic_vram_info, get_ram_usage, reset_vram_peak
from contextlib import contextmanager


class Debug:
    """
    Unified debug logging for generation pipeline and BlockSwap monitoring
    
    Features:
    - Structured logging with categories
    - Memory tracking (VRAM/RAM)
    - Timing utilities
    - BlockSwap operation tracking
    - Minimal overhead when disabled
    - Timestamped logs for better troubleshooting
    - Force parameters for critical logs
    """
    
    # Icon mapping for different categories
    CATEGORY_ICONS = {
        "general": "🔄",      # General operations/processing
        "timing": "⚡",       # Performance timing
        "memory": "📊",       # Memory usage tracking
        "cache": "💾",        # Cache operations
        "cleanup": "🧹",      # Cleanup operations
        "setup": "🔧",        # Configuration/setup
        "generation": "🎬",   # Generation process
        "dit": "🚀",        # Model loading/operations
        "blockswap": "🔀",    # BlockSwap operations
        "download": "📥",     # Download operations
        "success": "✅",      # Successful completion
        "warning": "⚠️",      # Warnings
        "error": "❌",        # Errors
        "info": "ℹ️",         # Statistics/info
        "tip" :"💡",           # Tip/suggestion
        "video": "📹",        # Video/sequence info
        "reuse": "♻️",        # Reusing/recycling
        "runner": "🏃",       # Runner operations
        "vae": "🎨",          # VAE operations
        "store": "📦",        # Storing
        "precision": "🎯",    # Precision
        "device": "🖥️",       # Device info
        "file": "📂",         # File operations
        "none" : "",
    }
    
    
    def __init__(self, enabled: bool = False, show_timestamps: bool = True):
        self.enabled = enabled
        self.show_timestamps = show_timestamps
        self.timers: Dict[str, float] = {}
        self.memory_checkpoints: List[Dict[str, Any]] = []
        self.max_checkpoints = 100
        self.timer_hierarchy: Dict[str, List[str]] = {}
        self.timer_durations: Dict[str, float] = {}
        self.timer_messages: Dict[str, str] = {} 
        self.swap_times: List[Dict[str, Any]] = []
        self.vram_history: List[float] = []
        self.active_timer_stack: List[str] = [] 
        self.timer_namespace: str = ""
        

    def log(self, message: str, level: str = "INFO", category: str = "general", force: bool = False) -> None:
        """
        Log a categorized message with optional timestamp
        
        Args:
            message: Message to log
            level: Log level (INFO, WARN, ERROR)
            category: Category for the message
            force: If True, always log regardless of enabled state (for critical messages)
        """
        # Always log forced messages or if debugging is enabled - early return if not
        if not (self.enabled or force):
            return
        
        # Get icon for category, fallback to general icon
        icon = self.CATEGORY_ICONS.get(category, self.CATEGORY_ICONS["general"])
        
        # Format prefix based on level
        if level == "WARNING":
            icon = self.CATEGORY_ICONS["warning"]
        elif level == "ERROR":
            icon = self.CATEGORY_ICONS["error"]
        
        # Build the log message with optional timestamp
        if self.show_timestamps:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            prefix = f"[{timestamp}] {icon}"
        else:
            prefix = f"{icon}"
        
        if level != "INFO":
            prefix += f" [{level}]"
        
        print(f"{prefix} {message}")
    

    @contextmanager
    def timer_context(self, namespace: str):
        """
        Context manager for setting a timer namespace temporarily.
        All timers started within this context will be prefixed with the namespace.
        
        Usage:
            with debug.timer_context("batch_1"):
                debug.start_timer("vae_encode")  # Will be "batch_1_vae_encode"
        """
        old_namespace = self.timer_namespace
        self.timer_namespace = namespace
        try:
            yield
        finally:
            self.timer_namespace = old_namespace
            

    def start_timer(self, name: str, force: bool = False) -> None:
        """
        Start a named timer
        
        Args:
            name: Timer name
            force: If True, start timer even when debug is disabled
        """
        if self.enabled or force:
            # Apply namespace if set
            if self.timer_namespace:
                name = f"{self.timer_namespace}_{name}"

            self.timers[name] = time.time()
            
            # Auto-hierarchy: if there's an active timer, this is a child
            if self.active_timer_stack:
                parent = self.active_timer_stack[-1]
                if parent not in self.timer_hierarchy:
                    self.timer_hierarchy[parent] = []
                # Only add if not already a child (prevents duplicates)
                if name not in self.timer_hierarchy[parent]:
                    self.timer_hierarchy[parent].append(name)
            
            # Push to stack
            self.active_timer_stack.append(name)
    

    def end_timer(self, name: str, message: Optional[str] = None, 
              force: bool = False, show_breakdown: bool = False,
              custom_children: Optional[Dict[str, float]] = None) -> float:
        """
        End a timer and optionally log its duration
        
        Args:
            name: Timer name
            message: Optional message to log with the duration
            force: If True, log even when debug is disabled (for critical timings)
            show_breakdown: If True, show breakdown of child timers
            custom_children: Optional dict of child timer names and durations to override automatic hierarchy
        
        Returns:
            Duration in seconds (0.0 if timer not found)
        """
        # Apply namespace if set
        if self.timer_namespace:
            name = f"{self.timer_namespace}_{name}"

        # Check if timer exists
        if name not in self.timers:
            return 0.0
        
        duration = time.time() - self.timers[name]
        self.timer_durations[name] = duration
        # Store the message for later use in summary
        if message:
            self.timer_messages[name] = message
        del self.timers[name]

        # Pop from stack if this is the current active timer
        if self.active_timer_stack and self.active_timer_stack[-1] == name:
            self.active_timer_stack.pop()

        # If debug is disabled and not forcing, return early
        if not self.enabled and not force:
            return duration
        
        # ONLY log if show_breakdown is True - this means it's a major summary timer
        if message and show_breakdown:
            # Use custom children if provided, otherwise use automatic hierarchy
            if custom_children:
                children = custom_children
                child_total = sum(children.values())
                unaccounted = duration - child_total
                
                self.log(f"{message}: {duration:.2f}s", category="timing", force=force)
                
                # Sort custom children by duration for better readability
                sorted_children = sorted(children.items(), key=lambda x: x[1], reverse=True)
                
                for child_name, child_duration in sorted_children:
                    if child_duration >= 0.01:  # Only show if >= 10ms
                        self.log(f"  └─ {child_name}: {child_duration:.2f}s", category="timing", force=force)
            else:
                # Use automatic hierarchy tracking
                children = self.timer_hierarchy.get(name, [])
                child_total = sum(self.timer_durations.get(child, 0) for child in children)
                unaccounted = duration - child_total
                
                self.log(f"{message}: {duration:.2f}s", category="timing", force=force)
                
                # Sort children by duration for better readability
                sorted_children = sorted(children, key=lambda c: self.timer_durations.get(c, 0), reverse=True)

                for child in sorted_children:
                    child_duration = self.timer_durations.get(child, 0)
                    if child_duration >= 0.01:  # Only show if >= 10ms
                        child_message = self.timer_messages.get(child, child)
                        self.log(f"  └─ {child_message}: {child_duration:.2f}s", category="timing", force=force)
                        
                        # Recursively show grandchildren
                        if child in self.timer_hierarchy:
                            grandchildren = self.timer_hierarchy[child]
                            sorted_grandchildren = sorted(grandchildren, key=lambda c: self.timer_durations.get(c, 0), reverse=True)
                            
                            for grandchild in sorted_grandchildren:
                                grandchild_duration = self.timer_durations.get(grandchild, 0)
                                if grandchild_duration >= 0.01:  # Only show if >= 10ms
                                    grandchild_message = self.timer_messages.get(grandchild, grandchild)
                                    self.log(f"      └─ {grandchild_message}: {grandchild_duration:.2f}s", category="timing", force=force)
            
            if unaccounted > 0.01:  # Show if more than 10ms unaccounted
                self.log(f"  └─ (other operations): {unaccounted:.2f}s", category="timing", force=force)
        
        return duration


    def log_memory_state(self, label: str, show_diff: bool = True, show_tensors: bool = False, 
                        detailed_tensors: bool = False, force: bool = False) -> None:
        """
        Log current memory state with minimal overhead.
        
        Args:
            label: Description for this checkpoint
            show_diff: Show change from last checkpoint
            show_tensors: Include tensor counts
            detailed_tensors: Show detailed tensor analysis (use sparingly)
            force: If True, always log regardless of enabled state
        """
        if not (self.enabled or force):
            return
        
        # Collect memory metrics efficiently
        memory_info = self._collect_memory_metrics()
        
        # Show category
        self.log(f"{label}:", category="memory", force=force)

        # Show VRAM
        if memory_info['summary_vram']:
            self.log(f"{memory_info['summary_vram']}", category="memory", force=force)

        # Show RAM
        if memory_info['summary_ram']:
            self.log(f"{memory_info['summary_ram']}", category="memory", force=force)

        # Show tensors
        if show_tensors:
            tensor_stats = self._collect_tensor_stats(detailed=detailed_tensors)
            self.log(f"{tensor_stats['summary']}", category="memory", force=force)
        
        # Show diff from last checkpoint
        if show_diff and self.memory_checkpoints:
            self._log_memory_diff(current_metrics=memory_info, force=force)

       # Log detailed analysis if requested
        if detailed_tensors and tensor_stats.get('details'):
            self._log_detailed_tensor_analysis(details=tensor_stats['details'], force=force)
                
        # Store checkpoint with memory limit
        self._store_checkpoint(label, memory_info)

        # Reset PyTorch's peak memory stats for next interval
        reset_vram_peak(debug=self)
    

    def _collect_memory_metrics(self) -> Dict[str, Any]:
        """Collect current memory metrics efficiently."""
        metrics = {
            'vram_allocated': 0.0,
            'vram_reserved': 0.0,
            'vram_free': 0.0,
            'vram_total': 0.0,
            'vram_peak_since_last': 0.0,
            'ram_process': 0.0,
            'ram_available': 0.0,
            'ram_total': 0.0,
            'ram_others': 0.0,
            'summary_vram': "",
            'summary_ram': ""
        }
        
        # VRAM metrics
        if torch.cuda.is_available():
            metrics['vram_allocated'], metrics['vram_reserved'], current_global_peak = get_vram_usage(debug=self)
            
            # Calculate peak since last log_memory_state
            # This captures the actual peak that occurred between calls
            metrics['vram_peak_since_last'] = current_global_peak
            
            vram_info = get_basic_vram_info()
            
            if "error" not in vram_info:
                metrics['vram_free'] = vram_info["free_gb"]
                metrics['vram_total'] = vram_info["total_gb"]
                
                metrics['summary_vram'] = (f"  [VRAM] {metrics['vram_allocated']:.2f}GB allocated / "
                        f"{metrics['vram_reserved']:.2f}GB reserved / "
                        f"Peak: {metrics['vram_peak_since_last']:.2f}GB / "
                        f"{metrics['vram_free']:.2f}GB free / "
                        f"{metrics['vram_total']:.2f}GB total")
            else:
                metrics['summary_vram'] = ""
        else:
            metrics['summary_vram'] = ""
        
        # RAM metrics using new function
        metrics['ram_process'], metrics['ram_available'], metrics['ram_total'], metrics['ram_others'] = get_ram_usage(debug=self)
        
        if metrics['ram_total'] > 0:
            metrics['summary_ram'] = (f"  [RAM] {metrics['ram_process']:.2f}GB process / "
                      f"{metrics['ram_others']:.2f}GB others / "
                      f"{metrics['ram_available']:.2f}GB free / "
                      f"{metrics['ram_total']:.2f}GB total")
        else:
            metrics['summary_ram'] = ""
        
        # Update VRAM history for tracking
        if torch.cuda.is_available():
            self.vram_history.append(metrics['vram_allocated'])
        
        return metrics
    

    def _collect_tensor_stats(self, detailed: bool = False) -> Dict[str, Any]:
        """Collect tensor statistics with minimal overhead."""
        stats = {
            'gpu_count': 0,
            'cpu_count': 0,
            'total_count': 0,
            'summary': "",
            'details': None
        }
        
        if detailed:
            stats['details'] = {
                'gpu_tensors': [],
                'large_cpu_tensors': [],
                'shape_patterns': {},
                'module_types': {}
            }
        
        # Single pass through gc objects
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    stats['total_count'] += 1
                    is_gpu = obj.is_cuda
                    
                    if is_gpu:
                        stats['gpu_count'] += 1
                    else:
                        stats['cpu_count'] += 1
                    
                    # Collect detailed info if requested
                    if detailed and obj.numel() > 0:
                        size_mb = obj.element_size() * obj.nelement() / (1024**2)
                        
                        if is_gpu or size_mb > 10:  # Only track GPU tensors or large CPU tensors
                            tensor_info = {
                                'shape': tuple(obj.shape),
                                'dtype': str(obj.dtype),
                                'size_mb': size_mb,
                                'requires_grad': obj.requires_grad
                            }
                            
                            if is_gpu:
                                stats['details']['gpu_tensors'].append(tensor_info)
                            elif size_mb > 10:  # Large CPU tensors (>10MB)
                                stats['details']['large_cpu_tensors'].append(tensor_info)
                            
                            # Track shape patterns
                            shape_key = str(tuple(obj.shape))
                            stats['details']['shape_patterns'][shape_key] = stats['details']['shape_patterns'].get(shape_key, 0) + 1
                
                elif detailed and isinstance(obj, torch.nn.Module):
                    module_type = type(obj).__name__
                    stats['details']['module_types'][module_type] = stats['details']['module_types'].get(module_type, 0) + 1
                        
            except (ReferenceError, AttributeError):
                # Object was deleted or doesn't have expected attributes
                pass
        
        stats['summary'] = f"  [Tensors] {stats['gpu_count']} GPU / {stats['cpu_count']} CPU / {stats['total_count']} total"
        
        return stats
    

    def _log_detailed_tensor_analysis(self, details: Dict[str, Any], force: bool = False) -> None:
        """Log detailed tensor analysis when requested."""
        
        # GPU tensors
        if details['gpu_tensors']:
            gpu_total_gb = sum(t['size_mb'] for t in details['gpu_tensors']) / 1024
            self.log(f"  GPU tensors: {len(details['gpu_tensors'])} using {gpu_total_gb:.2f}GB", category="memory", force=force)
            
            # Show top 5 largest
            largest = sorted(details['gpu_tensors'], key=lambda x: x['size_mb'], reverse=True)[:5]
            for t in largest:
                self.log(f"    {t['shape']}: {t['size_mb']:.2f}MB, {t['dtype']}", category="memory", force=force)
        
        # Large CPU tensors
        if details['large_cpu_tensors']:
            cpu_large_gb = sum(t['size_mb'] for t in details['large_cpu_tensors']) / 1024
            self.log(f"  Large CPU tensors (>10MB):", category="memory", force=force)
            self.log(f"    {len(details['large_cpu_tensors'])} using {cpu_large_gb:.2f}GB", category="memory", force=force)
            
            # Show top 3 largest
            largest = sorted(details['large_cpu_tensors'], key=lambda x: x['size_mb'], reverse=True)[:3]
            for t in largest:
                self.log(f"    {t['shape']}: {t['size_mb']:.2f}MB, {t['dtype']}", category="memory", force=force)
        
        # Common shape patterns
        if details['shape_patterns']:
            common_shapes = sorted(details['shape_patterns'].items(), 
                                  key=lambda x: x[1], reverse=True)[:5]
            if len(common_shapes) > 0:
                self.log("  Common tensor shapes:", category="memory", force=force)
                for shape, count in common_shapes:
                    if count > 1:
                        self.log(f"    {shape}: {count} instances", category="memory", force=force)
        
        # Module instances
        if details['module_types']:
            multi_instance = [(k, v) for k, v in details['module_types'].items() if v > 1]
            if multi_instance:
                self.log("  Multiple module instances:", category="memory", force=force)
                for mtype, count in sorted(multi_instance, key=lambda x: x[1], reverse=True)[:5]:
                    self.log(f"    {mtype}: {count} instances", category="memory", force=force)
    

    def _log_memory_diff(self, current_metrics: Dict[str, Any], force: bool = False) -> None:
        """Log memory changes from last checkpoint."""
        last = self.memory_checkpoints[-1]
        
        vram_diff = current_metrics['vram_allocated'] - last.get('vram_allocated', 0)
        ram_diff = current_metrics['ram_process'] - last.get('ram_process', 0)
        
        diffs = []
        if abs(vram_diff) > 0.01:
            sign = "+" if vram_diff > 0 else ""
            diffs.append(f"VRAM {sign}{vram_diff:.2f}GB")
        if abs(ram_diff) > 0.01:
            sign = "+" if ram_diff > 0 else ""
            diffs.append(f"RAM {sign}{ram_diff:.2f}GB")
        
        if diffs:
            self.log(f"  Memory changes: {', '.join(diffs)}", category="memory", force=force)
    

    def _store_checkpoint(self, label: str, metrics: Dict[str, Any]) -> None:
        """Store checkpoint with memory limit to prevent leaks."""
        checkpoint = {
            'label': label,
            'timestamp': time.time(),
            'vram_allocated': metrics['vram_allocated'],
            'vram_reserved': metrics['vram_reserved'],
            'vram_free': metrics['vram_free'],
            'ram_process': metrics['ram_process'],
            'ram_available': metrics['ram_available'],
            'ram_others': metrics['ram_others']
        }
        
        self.memory_checkpoints.append(checkpoint)
        
        # Prevent memory leak by limiting checkpoint history
        if len(self.memory_checkpoints) > self.max_checkpoints:
            # Keep first and last N/2 checkpoints for better history coverage
            mid = self.max_checkpoints // 2
            self.memory_checkpoints = (self.memory_checkpoints[:mid] + 
                                      self.memory_checkpoints[-mid:])
    

    def log_swap_time(self, component_id: Union[int, str], duration: float, 
                 component_type: str = "block", force: bool = False) -> None:
        """
        Log swap timing information for BlockSwap operations
        
        Args:
            component_id: Identifier for the component being swapped
            duration: Duration of the swap in seconds
            component_type: Type of component ('block' or other)
            force: If True, always log regardless of enabled state
        """
        if self.enabled or force:
            # Store timing data
            self.swap_times.append({
                'component_id': component_id,
                'component_type': component_type,
                'duration': duration,
            })
            
            # Format message based on component type
            if component_type == "block":
                message = f"Block {component_id} swap: {duration*1000:.2f}ms"
            else:
                message = f"{component_type} {component_id} swap: {duration*1000:.2f}ms"
            
            self.log(message, category="blockswap", force=force)
    
    
    def get_swap_summary(self) -> Dict[str, Any]:
        """Get summary of swap operations for analysis"""
        if not self.swap_times:
            return {}
        
        # Group by component type
        block_swaps = [s for s in self.swap_times if s['component_type'] == 'block']
        io_swaps = [s for s in self.swap_times if s['component_type'] != 'block']
        
        # Calculate statistics
        summary = {
            'total_swaps': len(self.swap_times),
            'block_swaps': len(block_swaps),
            'io_swaps': len(io_swaps),
        }
        
        if block_swaps:
            block_times = [s['duration'] for s in block_swaps]
            summary['block_avg_ms'] = sum(block_times) * 1000 / len(block_times)
            summary['block_total_ms'] = sum(block_times) * 1000
            summary['block_min_ms'] = min(block_times) * 1000
            summary['block_max_ms'] = max(block_times) * 1000
            
            # Track which blocks are swapped most frequently
            block_frequency = {}
            for swap in block_swaps:
                block_id = swap['component_id']
                block_frequency[block_id] = block_frequency.get(block_id, 0) + 1
            summary['most_swapped_block'] = max(block_frequency, key=block_frequency.get)
            summary['most_swapped_count'] = block_frequency[summary['most_swapped_block']]
        
        if io_swaps:
            io_times = [s['duration'] for s in io_swaps]
            summary['io_avg_ms'] = sum(io_times) * 1000 / len(io_times)
            summary['io_total_ms'] = sum(io_times) * 1000
            
            # Track which I/O components are swapped
            io_components = list(set(s['component_id'] for s in io_swaps))
            summary['io_components_swapped'] = io_components
        
        # VRAM efficiency metrics
        if self.vram_history:
            summary['peak_vram_gb'] = max(self.vram_history)
            summary['avg_vram_gb'] = sum(self.vram_history) / len(self.vram_history)
            summary['vram_variation_gb'] = max(self.vram_history) - min(self.vram_history)
        
        return summary
    
    
    def clear_history(self) -> None:
        """Clear all history tracking"""
        self.timers.clear()
        self.memory_checkpoints.clear()
        self.swap_times.clear()
        self.vram_history.clear()
        self.timer_hierarchy.clear()
        self.timer_durations.clear()
        self.timer_messages.clear()
        self.active_timer_stack.clear()