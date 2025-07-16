# Owner(s): ["module: optimizer"]
from torch.testing._internal.common_utils import TestCase, instantiate_parametrized_tests, load_tests
import torch
from torch.optim import Muon

load_tests = load_tests


def rosenbrock(tensor):
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


class TestMuon(TestCase):
    def test_basic_convergence(self):
        param = torch.nn.Parameter(torch.tensor([[0.8, 0.7]], dtype=torch.float32))
        optim = Muon([param], lr=0.1)
        for _ in range(50):
            optim.zero_grad()
            loss = rosenbrock(param.view(-1))
            loss.backward()
            optim.step()
        self.assertLess(loss.item(), 1.0)


instantiate_parametrized_tests(TestMuon)

if __name__ == "__main__":
    print("These tests should be run through test/test_optim.py instead")
