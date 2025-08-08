class MetaLearningAdapter:
    """
    Quickly adapt to new subjects using meta-learning
    """
    
    def __init__(self, base_model):
        self.base_model = base_model
        self.meta_optimizer = torch.optim.Adam(
            self.base_model.parameters(), 
            lr=0.001
        )
        
    def maml_adaptation(self, support_set, query_set, inner_steps=5):
        """
        Model-Agnostic Meta-Learning for quick subject adaptation
        """
        # Clone model for inner loop
        adapted_model = self.clone_model()
        inner_optimizer = torch.optim.SGD(
            adapted_model.parameters(), 
            lr=0.01
        )
        
        # Inner loop: adapt on support set
        for _ in range(inner_steps):
            support_loss = self.compute_loss(
                adapted_model, 
                support_set
            )
            inner_optimizer.zero_grad()
            support_loss.backward()
            inner_optimizer.step()
            
        # Outer loop: evaluate on query set
        query_loss = self.compute_loss(adapted_model, query_set)
        
        # Update meta-parameters
        self.meta_optimizer.zero_grad()
        query_loss.backward()
        self.meta_optimizer.step()
        
        return adapted_model