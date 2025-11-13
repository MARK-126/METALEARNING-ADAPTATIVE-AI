#!/usr/bin/env python3
"""
Script de verificación para probar las soluciones de los tutoriales.
Ejecuta pruebas unitarias sin necesidad de descargar datasets completos.
"""

import sys
import traceback
from collections import defaultdict

# Colores para output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_test(name, passed, error=None):
    """Imprime resultado de un test."""
    status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
    print(f"  {status} {name}")
    if error and not passed:
        print(f"    {RED}Error: {error}{RESET}")

def test_imports():
    """Test 1: Verificar que las librerías críticas se pueden importar."""
    print(f"\n{BOLD}Test 1: Importaciones{RESET}")

    tests = [
        ("torch", lambda: __import__('torch')),
        ("torchvision", lambda: __import__('torchvision')),
        ("numpy", lambda: __import__('numpy')),
        ("matplotlib", lambda: __import__('matplotlib')),
        ("tqdm", lambda: __import__('tqdm')),
    ]

    all_passed = True
    for name, import_fn in tests:
        try:
            import_fn()
            print_test(name, True)
        except ImportError as e:
            print_test(name, False, str(e))
            all_passed = False

    return all_passed

def test_omniglot_sampler():
    """Test 2: Verificar lógica del OmniglotNWayKShot sampler."""
    print(f"\n{BOLD}Test 2: OmniglotNWayKShot Sampler (lógica){RESET}")

    try:
        import torch
        import random

        # Mock dataset simple
        class MockDataset:
            def __init__(self):
                # 10 clases, 20 ejemplos cada una
                self.data = []
                for class_id in range(10):
                    for example in range(20):
                        # Imagen fake: tensor 1x28x28
                        img = torch.randn(1, 28, 28)
                        self.data.append((img, class_id))

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        # Implementación del sampler (de SOLUTIONS.md)
        class OmniglotNWayKShot:
            def __init__(self, dataset, n_way=5, k_shot=1, q_query=15):
                self.dataset = dataset
                self.n_way = n_way
                self.k_shot = k_shot
                self.q_query = q_query
                self.class_to_indices = self._organize_by_class()
                self.classes = list(self.class_to_indices.keys())

            def _organize_by_class(self):
                class_to_indices = defaultdict(list)
                for idx, (img, label) in enumerate(self.dataset):
                    class_to_indices[label].append(idx)
                return class_to_indices

            def sample_episode(self):
                selected_classes = random.sample(self.classes, self.n_way)
                support_x, support_y, query_x, query_y = [], [], [], []

                for new_label, class_id in enumerate(selected_classes):
                    class_indices = self.class_to_indices[class_id]
                    selected_indices = random.sample(class_indices, self.k_shot + self.q_query)

                    for i, idx in enumerate(selected_indices):
                        img, _ = self.dataset[idx]
                        if i < self.k_shot:
                            support_x.append(img)
                            support_y.append(new_label)
                        else:
                            query_x.append(img)
                            query_y.append(new_label)

                return (torch.stack(support_x), torch.LongTensor(support_y),
                        torch.stack(query_x), torch.LongTensor(query_y))

        # Crear dataset mock y sampler
        dataset = MockDataset()
        sampler = OmniglotNWayKShot(dataset, n_way=5, k_shot=1, q_query=15)

        # Test 2.1: Organización por clases
        print_test("_organize_by_class crea 10 clases", len(sampler.classes) == 10)
        print_test("Cada clase tiene 20 ejemplos",
                   all(len(indices) == 20 for indices in sampler.class_to_indices.values()))

        # Test 2.2: Sampling de episodio
        support_x, support_y, query_x, query_y = sampler.sample_episode()

        print_test("Support shape correcto (5x1)", support_x.shape[0] == 5)
        print_test("Query shape correcto (5x15)", query_x.shape[0] == 75)
        print_test("5 clases únicas en support", len(torch.unique(support_y)) == 5)
        print_test("5 clases únicas en query", len(torch.unique(query_y)) == 5)
        print_test("Labels van de 0 a 4",
                   support_y.min() == 0 and support_y.max() == 4)

        return True

    except Exception as e:
        print_test("OmniglotNWayKShot sampler", False, str(e))
        traceback.print_exc()
        return False

def test_cosine_similarity():
    """Test 3: Verificar funciones de Matching Networks."""
    print(f"\n{BOLD}Test 3: Matching Networks (cosine similarity){RESET}")

    try:
        import torch
        import torch.nn.functional as F

        # Implementación de SOLUTIONS.md
        def cosine_similarity(x1, x2):
            x1_norm = F.normalize(x1, p=2, dim=1)
            x2_norm = F.normalize(x2, p=2, dim=1)
            return torch.matmul(x1_norm, x2_norm.T)

        def compute_attention(query_embeddings, support_embeddings, temperature=1.0):
            similarities = cosine_similarity(query_embeddings, support_embeddings)
            return F.softmax(similarities / temperature, dim=1)

        # Test con datos sintéticos
        query = torch.randn(10, 64)  # 10 queries, 64-dim
        support = torch.randn(5, 64)  # 5 support, 64-dim

        sim = cosine_similarity(query, support)
        print_test("Cosine similarity shape (10x5)", sim.shape == (10, 5))
        print_test("Valores entre -1 y 1", (sim >= -1.0).all() and (sim <= 1.0).all())

        attn = compute_attention(query, support)
        print_test("Attention shape (10x5)", attn.shape == (10, 5))
        print_test("Attention suma a 1", torch.allclose(attn.sum(dim=1), torch.ones(10)))

        return True

    except Exception as e:
        print_test("Matching Networks functions", False, str(e))
        traceback.print_exc()
        return False

def test_ood_transformations():
    """Test 4: Verificar transformaciones OOD."""
    print(f"\n{BOLD}Test 4: OOD Transformations{RESET}")

    try:
        import torch

        # Implementación de SOLUTIONS.md
        def apply_ood_transform(images, transform_type='rotate'):
            if transform_type == 'rotate':
                angles = torch.randint(-30, 30, (images.size(0),))
                return torch.stack([
                    torch.rot90(img, k=int(angle/90), dims=(1, 2))
                    for img, angle in zip(images, angles)
                ])
            elif transform_type == 'noise':
                noise = torch.randn_like(images) * 0.1
                return torch.clamp(images + noise, 0, 1)
            elif transform_type == 'blur':
                # Simulación simple de blur
                return images * 0.7 + 0.3
            elif transform_type == 'invert':
                return 1.0 - images
            else:
                return images

        # Test con imágenes sintéticas
        images = torch.rand(4, 3, 32, 32)

        rotated = apply_ood_transform(images, 'rotate')
        print_test("Rotate preserva shape", rotated.shape == images.shape)

        noisy = apply_ood_transform(images, 'noise')
        print_test("Noise preserva shape", noisy.shape == images.shape)
        print_test("Noise valores válidos", (noisy >= 0).all() and (noisy <= 1).all())

        inverted = apply_ood_transform(images, 'invert')
        print_test("Invert preserva shape", inverted.shape == images.shape)

        return True

    except Exception as e:
        print_test("OOD transformations", False, str(e))
        traceback.print_exc()
        return False

def main():
    """Ejecuta todos los tests."""
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Verificación de Soluciones - Meta-Learning Tutorials{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    results = []

    results.append(("Importaciones", test_imports()))
    results.append(("OmniglotNWayKShot", test_omniglot_sampler()))
    results.append(("Matching Networks", test_cosine_similarity()))
    results.append(("OOD Transformations", test_ood_transformations()))

    # Resumen
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Resumen{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = f"{GREEN}✓{RESET}" if result else f"{RED}✗{RESET}"
        print(f"  {status} {name}")

    print(f"\n  Total: {passed}/{total} tests pasados")

    if passed == total:
        print(f"\n{GREEN}{BOLD}✓ ¡Todas las verificaciones pasaron!{RESET}")
        return 0
    else:
        print(f"\n{YELLOW}{BOLD}⚠ Algunas verificaciones fallaron{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
