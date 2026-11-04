# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Callback utilities (YOLOv5 风格的事件回调框架)."""

import inspect


class Callbacks:
    """Manages all registered callbacks (hook) functions.

    支持的事件：
        on_pretrain_routine_start / on_pretrain_routine_end
        on_train_start / on_train_epoch_start / on_train_batch_start / on_train_batch_end
        on_train_epoch_end / on_fit_epoch_end / on_model_save / on_train_end
        on_val_start / on_val_batch_start / on_val_batch_end / on_val_end
        on_predict_start / on_predict_batch_start / on_predict_postprocess_end / on_predict_end
    """

    def __init__(self):
        """Initialize the callbacks with default hooks."""
        self.valid_actions = [
            "on_pretrain_routine_start",
            "on_pretrain_routine_end",
            "on_train_start",
            "on_train_epoch_start",
            "on_train_batch_start",
            "optimizer_step",
            "on_train_batch_end",
            "on_train_epoch_end",
            "on_fit_epoch_end",
            "on_model_save",
            "on_train_end",
            "on_val_start",
            "on_val_batch_start",
            "on_val_batch_end",
            "on_val_end",
            "on_predict_start",
            "on_predict_batch_start",
            "on_predict_postprocess_end",
            "on_predict_end",
        ]
        self.callbacks = {k: [] for k in self.valid_actions}
        self.stop_training = False

    def register_action(self, hook, name="", callback=None):
        """Register a callback function to a specified hook."""
        assert hook in self.callbacks, f"hook {hook} is not registered!"
        assert callback is not None or name, "Either callback or name must be provided"
        if callback is not None:
            assert callable(callback), "callback must be a function"
            self.callbacks[hook].append({"name": name or callback.__name__, "callback": callback})
        else:
            def decorator(func):
                self.callbacks[hook].append({"name": name or func.__name__, "callback": func})
                return func

            return decorator

    def get_registered_actions(self, hook=None):
        """Return all registered callbacks, or those of one hook."""
        return self.callbacks[hook] if hook else self.callbacks

    def run(self, hook, *args, **kwargs):
        """Run all callbacks registered to `hook`."""
        assert hook in self.callbacks, f"hook {hook} is not registered!"
        for cb in self.callbacks[hook]:
            func = cb["callback"]
            # 只传函数声明中存在的参数，避免签名不匹配
            sig = inspect.signature(func)
            accepted = {k: v for k, v in kwargs.items() if k in sig.parameters or any(
                p.kind == p.VAR_KEYWORD for p in sig.parameters.values())}
            ret = func(*args, **accepted)
            if ret is not None and isinstance(ret, bool) and ret:
                self.stop_training = True

    def _log_hooks(self):
        """Return a dict of hook -> list of registered callback names."""
        return {k: [x["name"] for x in v] for k, v in self.callbacks.items() if v}
