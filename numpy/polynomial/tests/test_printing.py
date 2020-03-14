import pytest
from numpy.core import arange
import numpy.polynomial as poly
from numpy.testing import assert_equal


class TestStr:
    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0x¹ + 3.0x²"),
        ([-1, 0, 3, -1], "-1.0 + 0.0x¹ + 3.0x² - 1.0x³"),
        (arange(12), ("0.0 + 1.0x¹ + 2.0x² + 3.0x³ + 4.0x⁴ + 5.0x⁵ + 6.0x⁶ "
                      "+ 7.0x⁷ + 8.0x⁸ +\n9.0x⁹ + 10.0x¹⁰ + 11.0x¹¹")),
    ))
    def test_polynomial_str(self, inp, tgt):
        res = str(poly.Polynomial(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0T₁(x) + 3.0T₂(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0T₁(x) + 3.0T₂(x) - 1.0T₃(x)"),
        (arange(12), ("0.0 + 1.0T₁(x) + 2.0T₂(x) + 3.0T₃(x) + 4.0T₄(x) + "
                      "5.0T₅(x) + 6.0T₆(x) +\n7.0T₇(x) + 8.0T₈(x) + "
                      "9.0T₉(x) + 10.0T₁₀(x) + 11.0T₁₁(x)")),
    ))
    def test_chebyshev_str(self, inp, tgt):
        res = str(poly.Chebyshev(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0P₁(x) + 3.0P₂(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0P₁(x) + 3.0P₂(x) - 1.0P₃(x)"),
        (arange(12), ("0.0 + 1.0P₁(x) + 2.0P₂(x) + 3.0P₃(x) + 4.0P₄(x) + "
                      "5.0P₅(x) + 6.0P₆(x) +\n7.0P₇(x) + 8.0P₈(x) + "
                      "9.0P₉(x) + 10.0P₁₀(x) + 11.0P₁₁(x)")),
    ))
    def test_legendre_str(self, inp, tgt):
        res = str(poly.Legendre(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0H₁(x) + 3.0H₂(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0H₁(x) + 3.0H₂(x) - 1.0H₃(x)"),
        (arange(12), ("0.0 + 1.0H₁(x) + 2.0H₂(x) + 3.0H₃(x) + 4.0H₄(x) + "
                      "5.0H₅(x) + 6.0H₆(x) +\n7.0H₇(x) + 8.0H₈(x) + "
                      "9.0H₉(x) + 10.0H₁₀(x) + 11.0H₁₁(x)")),
    ))
    def test_hermite_str(self, inp, tgt):
        res = str(poly.Hermite(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0He₁(x) + 3.0He₂(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0He₁(x) + 3.0He₂(x) - 1.0He₃(x)"),
        (arange(12), ("0.0 + 1.0He₁(x) + 2.0He₂(x) + 3.0He₃(x) + 4.0He₄(x) + "
                      "5.0He₅(x) +\n6.0He₆(x) + 7.0He₇(x) + 8.0He₈(x) + "
                      "9.0He₉(x) + 10.0He₁₀(x) + 11.0He₁₁(x)")),
    ))
    def test_hermiteE_str(self, inp, tgt):
        res = str(poly.HermiteE(inp))
        assert_equal(res, tgt)

    @pytest.mark.parametrize(('inp', 'tgt'), (
        ([1, 2, 3], "1.0 + 2.0L₁(x) + 3.0L₂(x)"),
        ([-1, 0, 3, -1], "-1.0 + 0.0L₁(x) + 3.0L₂(x) - 1.0L₃(x)"),
        (arange(12), ("0.0 + 1.0L₁(x) + 2.0L₂(x) + 3.0L₃(x) + 4.0L₄(x) + "
                      "5.0L₅(x) + 6.0L₆(x) +\n7.0L₇(x) + 8.0L₈(x) + "
                      "9.0L₉(x) + 10.0L₁₀(x) + 11.0L₁₁(x)")),
    ))
    def test_laguerre_str(self, inp, tgt):
        res = str(poly.Laguerre(inp))
        assert_equal(res, tgt)

class TestRepr:
    def test_polynomial_str(self):
        res = repr(poly.Polynomial([0, 1]))
        tgt = 'Polynomial([0., 1.], domain=[-1,  1], window=[-1,  1])'
        assert_equal(res, tgt)

    def test_chebyshev_str(self):
        res = repr(poly.Chebyshev([0, 1]))
        tgt = 'Chebyshev([0., 1.], domain=[-1,  1], window=[-1,  1])'
        assert_equal(res, tgt)

    def test_legendre_repr(self):
        res = repr(poly.Legendre([0, 1]))
        tgt = 'Legendre([0., 1.], domain=[-1,  1], window=[-1,  1])'
        assert_equal(res, tgt)

    def test_hermite_repr(self):
        res = repr(poly.Hermite([0, 1]))
        tgt = 'Hermite([0., 1.], domain=[-1,  1], window=[-1,  1])'
        assert_equal(res, tgt)

    def test_hermiteE_repr(self):
        res = repr(poly.HermiteE([0, 1]))
        tgt = 'HermiteE([0., 1.], domain=[-1,  1], window=[-1,  1])'
        assert_equal(res, tgt)

    def test_laguerre_repr(self):
        res = repr(poly.Laguerre([0, 1]))
        tgt = 'Laguerre([0., 1.], domain=[0, 1], window=[0, 1])'
        assert_equal(res, tgt)
