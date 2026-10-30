"""The physics hash keys sewing's source by its code only: documentation, formatting, comments and message
text leave the key unchanged, and a functional edit changes it."""

import unittest

from sewing import env as E


class CodeKeyTest(unittest.TestCase):
    SRC = ('"""Module doc."""\n'
           'def f(x):\n'
           '    """Doc."""\n'
           '    if x < 0:\n'
           '        raise ValueError(f"negative: {x}")\n'
           '    return 2*x  # twice\n')

    def test_documentation_formatting_and_messages_are_not_code(self):
        k = E._code_key(self.SRC)
        same = ('"""Another module doc."""\n\n\n'
                'def f(x):\n'
                '    """Another doc, longer."""\n'
                '    # a comment\n'
                '    if x < 0:\n'
                '        raise ValueError("x was negative; see the manual: " + str(x))\n'
                '    return 2 * x\n')
        self.assertEqual(k, E._code_key(same))

    def test_a_functional_edit_is_code(self):
        k = E._code_key(self.SRC)
        for edit in (self.SRC.replace("2*x", "3*x"), self.SRC.replace("x < 0", "x <= 0"),
                     self.SRC.replace("ValueError", "RuntimeError"), self.SRC + "Y = 1\n"):
            with self.subTest(edit=edit):
                self.assertNotEqual(k, E._code_key(edit))


if __name__ == "__main__":
    unittest.main()
