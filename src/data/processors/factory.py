from .spectral import SpectralProcessor
from .legacy import PCAOTProcessor

class ProcessorFactory:
    @staticmethod
    def get_processor(cfg):
        method = cfg.method.name
        print(f"⚙️ Initializing Processor: {method}")
        
        if method == "spectral":
            return SpectralProcessor(cfg)
        elif method == "alignment":
            # 这是原论文的 Baseline 逻辑
            return PCAOTProcessor(cfg)
        else:
            raise ValueError(f"Unknown method: {method}")