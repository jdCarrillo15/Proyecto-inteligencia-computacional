"""
Implementación de Focal Loss para manejo de desbalanceo de clases.
Focal Loss enfatiza ejemplos difíciles y reduce el peso de ejemplos bien clasificados.
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


class FocalLoss(keras.losses.Loss):
    """
    Focal Loss para clasificación multiclase.
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Donde:
        - p_t es la probabilidad del modelo para la clase correcta
        - alpha: factor de balance para clases (similar a class_weights)
        - gamma: factor de enfoque, reduce la pérdida para ejemplos bien clasificados
    
    Parámetros recomendados:
        - gamma=2.0: Enfoca en ejemplos difíciles
        - alpha=None: Se puede usar con class_weights
    
    Referencias:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
        https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, gamma=2.0, alpha=None, name='focal_loss', **kwargs):
        """
        Inicializa Focal Loss.
        
        Args:
            gamma: Factor de enfoque (>=0). Valores típicos: [0.5, 5.0]
                   gamma=0 equivale a categorical crossentropy
                   gamma=2.0 es el valor recomendado por el paper
            alpha: Tensor de pesos por clase o None. Si None, sin balance adicional.
                   Debe tener forma (num_classes,)
            name: Nombre de la pérdida
        """
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        
    def call(self, y_true, y_pred):
        """
        Calcula Focal Loss.
        
        Args:
            y_true: Ground truth (one-hot encoded), shape (batch_size, num_classes)
            y_pred: Predicciones del modelo (probabilidades), shape (batch_size, num_classes)
            
        Returns:
            Focal loss escalar
        """
        # Clip predictions para estabilidad numérica
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calcular cross entropy
        cross_entropy = -y_true * K.log(y_pred)
        
        # Calcular focal loss
        # (1 - p_t)^gamma donde p_t es la probabilidad de la clase correcta
        focal_weight = K.pow(1.0 - y_pred, self.gamma)
        focal_loss = focal_weight * cross_entropy
        
        # Aplicar alpha (class weights) si está especificado
        if self.alpha is not None:
            alpha_weight = y_true * self.alpha
            focal_loss = alpha_weight * focal_loss
        
        # Retornar la pérdida promedio
        return K.mean(K.sum(focal_loss, axis=-1))
    
    def get_config(self):
        """Retorna configuración para serialización."""
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha if self.alpha is None else self.alpha.numpy().tolist()
        })
        return config


def categorical_focal_loss(gamma=2.0, alpha=None):
    """
    Factory function para crear Focal Loss con parámetros específicos.
    
    Args:
        gamma: Factor de enfoque
        alpha: Pesos por clase (array-like o None)
        
    Returns:
        Función de pérdida compatible con Keras
        
    Ejemplo:
        model.compile(
            optimizer='adam',
            loss=categorical_focal_loss(gamma=2.0),
            metrics=['accuracy']
        )
    """
    if alpha is not None and not isinstance(alpha, tf.Tensor):
        alpha = tf.constant(alpha, dtype=tf.float32)
    
    focal = FocalLoss(gamma=gamma, alpha=alpha)
    
    def loss(y_true, y_pred):
        return focal(y_true, y_pred)
    
    return loss


class BinaryFocalLoss(keras.losses.Loss):
    """
    Focal Loss para clasificación binaria.
    Útil si en el futuro se necesita detección binaria (enfermo/sano).
    """
    
    def __init__(self, gamma=2.0, alpha=0.25, name='binary_focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        """Calcula binary focal loss."""
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Binary cross entropy
        bce = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        
        # Focal weight
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = K.pow(1.0 - p_t, self.gamma)
        
        # Alpha balancing
        alpha_weight = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        
        focal_loss = alpha_weight * focal_weight * bce
        
        return K.mean(focal_loss)


def test_focal_loss():
    """Función de prueba para verificar que Focal Loss funciona correctamente."""
    import numpy as np
    
    print("=" * 80)
    print("TESTING FOCAL LOSS IMPLEMENTATION")
    print("=" * 80)
    
    # Crear datos de prueba
    batch_size = 32
    num_classes = 15
    
    # Ground truth (one-hot)
    y_true = np.zeros((batch_size, num_classes))
    y_true[np.arange(batch_size), np.random.randint(0, num_classes, batch_size)] = 1
    y_true = tf.constant(y_true, dtype=tf.float32)
    
    # Predicciones (softmax)
    y_pred = tf.nn.softmax(tf.random.normal((batch_size, num_classes)))
    
    # Test 1: Focal Loss sin alpha
    print("\n1. Testing Focal Loss (gamma=2.0, no alpha):")
    focal_loss = FocalLoss(gamma=2.0)
    loss_value = focal_loss(y_true, y_pred)
    print(f"   Loss value: {loss_value.numpy():.4f}")
    
    # Test 2: Focal Loss con alpha (class weights)
    print("\n2. Testing Focal Loss (gamma=2.0, with alpha):")
    alpha = tf.constant([1.0] * num_classes, dtype=tf.float32)
    focal_loss_alpha = FocalLoss(gamma=2.0, alpha=alpha)
    loss_value_alpha = focal_loss_alpha(y_true, y_pred)
    print(f"   Loss value: {loss_value_alpha.numpy():.4f}")
    
    # Test 3: Comparación con categorical crossentropy
    print("\n3. Comparing with Categorical Crossentropy:")
    cce = keras.losses.CategoricalCrossentropy()
    cce_value = cce(y_true, y_pred)
    print(f"   CCE value: {cce_value.numpy():.4f}")
    print(f"   Focal Loss value: {loss_value.numpy():.4f}")
    print(f"   Focal Loss emphasizes harder examples (should be different)")
    
    # Test 4: Verificar que gamma=0 ≈ categorical crossentropy
    print("\n4. Testing gamma=0 (should be similar to CCE):")
    focal_loss_0 = FocalLoss(gamma=0.0)
    loss_value_0 = focal_loss_0(y_true, y_pred)
    print(f"   Focal Loss (gamma=0): {loss_value_0.numpy():.4f}")
    print(f"   CCE: {cce_value.numpy():.4f}")
    print(f"   Difference: {abs(loss_value_0.numpy() - cce_value.numpy()):.6f}")
    
    # Test 5: Diferentes valores de gamma
    print("\n5. Testing different gamma values:")
    for gamma_val in [0.5, 1.0, 2.0, 3.0, 5.0]:
        fl = FocalLoss(gamma=gamma_val)
        loss = fl(y_true, y_pred)
        print(f"   Gamma={gamma_val}: Loss={loss.numpy():.4f}")
    
    # Test 6: Factory function
    print("\n6. Testing factory function:")
    loss_fn = categorical_focal_loss(gamma=2.0)
    loss_factory = loss_fn(y_true, y_pred)
    print(f"   Factory function loss: {loss_factory.numpy():.4f}")
    
    print("\n" + "=" * 80)
    print("✅ FOCAL LOSS TESTS COMPLETED")
    print("=" * 80)
    print("\nUsage in training:")
    print("  from utils.focal_loss import categorical_focal_loss")
    print("  model.compile(")
    print("      optimizer='adam',")
    print("      loss=categorical_focal_loss(gamma=2.0),")
    print("      metrics=['accuracy']")
    print("  )")


if __name__ == "__main__":
    test_focal_loss()
