from codecarbon import EmissionsTracker
import torch
import math
tracker = EmissionsTracker()
tracker.start()

# Training code...

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(2000):
    y_pred = a + b * x + c * x ** 2 + d * x ** 3
    loss = (y_pred - y).pow(2).sum()
    
    if t % 100 == 99:
        print(f'Epoch {t}, Loss: {loss.item():.4f}')
    
    loss.backward()
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: y = {a.item():.4f} + {b.item():.4f}x + {c.item():.4f}x² + {d.item():.4f}x³')

# # Your model/training code here
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # ... model definition, training loop ...

tracker.stop()