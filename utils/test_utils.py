"""
Utilidades para testing de ejercicios en los notebooks
"""

import sys
from typing import Callable, Any, List, Optional
import numpy as np
import torch


class Colors:
    """Códigos ANSI para colores en terminal"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_success(message: str = "¡Test pasado! ✅"):
    """Imprime mensaje de éxito"""
    print(f"{Colors.GREEN}{Colors.BOLD}{message}{Colors.END}")


def print_error(message: str):
    """Imprime mensaje de error"""
    print(f"{Colors.RED}{Colors.BOLD}❌ Error: {message}{Colors.END}")


def print_hint(hint_number: int, hint_text: str):
    """Imprime una pista numerada"""
    print(f"{Colors.CYAN}{Colors.BOLD}💡 Pista {hint_number}:{Colors.END} {hint_text}")


def print_warning(message: str):
    """Imprime mensaje de advertencia"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")


def run_test(test_func: Callable, test_name: str = "Test", verbose: bool = True) -> bool:
    """
    Ejecuta una función de test y maneja excepciones

    Args:
        test_func: Función de test a ejecutar
        test_name: Nombre del test
        verbose: Si True, imprime mensajes detallados

    Returns:
        True si el test pasó, False en caso contrario
    """
    try:
        test_func()
        if verbose:
            print_success(f"✅ {test_name} pasado!")
        return True
    except AssertionError as e:
        if verbose:
            print_error(f"{test_name} fallido: {str(e)}")
        return False
    except Exception as e:
        if verbose:
            print_error(f"{test_name} generó una excepción: {type(e).__name__}: {str(e)}")
        return False


def check_implementation(
    func: Callable,
    test_cases: List[tuple],
    test_name: str = "Implementación",
    tolerance: float = 1e-5
) -> bool:
    """
    Verifica que una función produzca los resultados esperados

    Args:
        func: Función a testear
        test_cases: Lista de tuplas (inputs, expected_output)
        test_name: Nombre del test
        tolerance: Tolerancia para comparaciones numéricas

    Returns:
        True si todos los tests pasan
    """
    all_passed = True

    for i, (inputs, expected) in enumerate(test_cases, 1):
        try:
            # Manejar inputs que son tuplas vs valores únicos
            if isinstance(inputs, tuple):
                result = func(*inputs)
            else:
                result = func(inputs)

            # Comparar resultados
            if isinstance(expected, (np.ndarray, torch.Tensor)):
                if isinstance(result, torch.Tensor):
                    result = result.detach().cpu().numpy()
                if isinstance(expected, torch.Tensor):
                    expected = expected.detach().cpu().numpy()

                if not np.allclose(result, expected, atol=tolerance):
                    print_error(f"{test_name} - Caso {i}: Resultado incorrecto")
                    print(f"  Esperado: {expected}")
                    print(f"  Obtenido: {result}")
                    all_passed = False
            else:
                if abs(result - expected) > tolerance:
                    print_error(f"{test_name} - Caso {i}: Resultado incorrecto")
                    print(f"  Esperado: {expected}")
                    print(f"  Obtenido: {result}")
                    all_passed = False

        except Exception as e:
            print_error(f"{test_name} - Caso {i}: Excepción - {type(e).__name__}: {str(e)}")
            all_passed = False

    if all_passed:
        print_success(f"✅ {test_name}: Todos los tests pasaron!")

    return all_passed


def check_model_output_shape(
    model: torch.nn.Module,
    input_shape: tuple,
    expected_output_shape: tuple,
    test_name: str = "Forma del output"
) -> bool:
    """
    Verifica que un modelo de PyTorch produzca la forma de output correcta

    Args:
        model: Modelo de PyTorch
        input_shape: Forma del input (batch_size, ...)
        expected_output_shape: Forma esperada del output
        test_name: Nombre del test

    Returns:
        True si la forma es correcta
    """
    try:
        dummy_input = torch.randn(*input_shape)
        with torch.no_grad():
            output = model(dummy_input)

        if output.shape != expected_output_shape:
            print_error(f"{test_name}: Forma incorrecta")
            print(f"  Esperado: {expected_output_shape}")
            print(f"  Obtenido: {output.shape}")
            return False

        print_success(f"✅ {test_name}: Forma correcta {output.shape}")
        return True

    except Exception as e:
        print_error(f"{test_name}: Excepción - {type(e).__name__}: {str(e)}")
        return False


def check_gradient_flow(
    model: torch.nn.Module,
    loss: torch.Tensor,
    test_name: str = "Flujo de gradientes"
) -> bool:
    """
    Verifica que los gradientes fluyan correctamente en un modelo

    Args:
        model: Modelo de PyTorch
        loss: Tensor de pérdida después de backward()
        test_name: Nombre del test

    Returns:
        True si hay flujo de gradientes
    """
    has_gradients = False
    no_gradient_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
                has_gradients = True
            else:
                no_gradient_params.append(name)

    if not has_gradients:
        print_error(f"{test_name}: No hay flujo de gradientes")
        return False

    if no_gradient_params:
        print_warning(f"{test_name}: Los siguientes parámetros no tienen gradientes: {no_gradient_params}")
    else:
        print_success(f"✅ {test_name}: Gradientes fluyendo correctamente")

    return True


class HintSystem:
    """Sistema de pistas progresivas para ejercicios"""

    def __init__(self, hints: List[str]):
        """
        Args:
            hints: Lista de pistas en orden de menos a más explícitas
        """
        self.hints = hints
        self.current_hint = 0

    def show_hint(self):
        """Muestra la siguiente pista"""
        if self.current_hint < len(self.hints):
            print_hint(self.current_hint + 1, self.hints[self.current_hint])
            self.current_hint += 1
        else:
            print(f"{Colors.MAGENTA}No hay más pistas disponibles. ¡Intenta revisar la documentación o preguntar!{Colors.END}")

    def reset(self):
        """Reinicia el sistema de pistas"""
        self.current_hint = 0

    def show_all(self):
        """Muestra todas las pistas"""
        for i, hint in enumerate(self.hints, 1):
            print_hint(i, hint)


def compare_tensors(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    tolerance: float = 1e-5,
    name: str = "Tensores"
) -> bool:
    """
    Compara dos tensores con tolerancia

    Args:
        tensor1: Primer tensor
        tensor2: Segundo tensor
        tolerance: Tolerancia para la comparación
        name: Nombre para los mensajes

    Returns:
        True si son iguales dentro de la tolerancia
    """
    try:
        if tensor1.shape != tensor2.shape:
            print_error(f"{name}: Formas diferentes - {tensor1.shape} vs {tensor2.shape}")
            return False

        if not torch.allclose(tensor1, tensor2, atol=tolerance):
            diff = torch.abs(tensor1 - tensor2).max().item()
            print_error(f"{name}: Valores diferentes (máxima diferencia: {diff})")
            return False

        print_success(f"✅ {name}: Idénticos dentro de la tolerancia")
        return True

    except Exception as e:
        print_error(f"{name}: Error en comparación - {type(e).__name__}: {str(e)}")
        return False
