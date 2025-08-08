class UncertaintyAwareDecoder:
    """
    Use ensemble of models with different synthetic augmentations
    to estimate uncertainty and improve robustness
    """
    
    def __init__(self, num_models=5):
        self.ensemble = [
            ContrastiveNeuralDecoder() 
            for _ in range(num_models)
        ]
        
    def decode_with_uncertainty(self, neural_signals):
        """
        Decode using ensemble and return uncertainty estimates
        """
        predictions = []
        
        for model in self.ensemble:
            # Each model trained with different synthetic augmentations
            pred = model(neural_signals, training=False)[0]
            predictions.append(torch.softmax(pred, dim=-1))
            
        # Stack predictions
        predictions = torch.stack(predictions)
        
        # Mean prediction
        mean_pred = predictions.mean(dim=0)
        
        # Uncertainty (entropy of ensemble disagreement)
        uncertainty = -torch.sum(
            mean_pred * torch.log(mean_pred + 1e-8), 
            dim=-1
        )
        
        return mean_pred, uncertainty
    
    def adaptive_beam_search(self, neural_signals, beam_width=5):
        """
        Use uncertainty to adaptively adjust beam search width
        """
        mean_pred, uncertainty = self.decode_with_uncertainty(neural_signals)
        
        # Increase beam width for uncertain predictions
        adaptive_width = beam_width * (1 + uncertainty.mean().item())
        adaptive_width = int(min(adaptive_width, beam_width * 3))
        
        return self.beam_search_decode(mean_pred, adaptive_width)