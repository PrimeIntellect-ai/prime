def build_metrics_string(metrics: dict[str, any], whitelist_keys: set[str]) -> str:
    metrics_string = ""
    for k, v in metrics.items():
        if k not in whitelist_keys:
            continue
        if metrics_string != '':
            metrics_string += ', '
        if isinstance(v, float):
            metrics_string += f'{k}: {v:.3f}'
        else:
            metrics_string += f'{k}: {v}'
    return metrics_string
