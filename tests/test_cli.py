from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from e38_w4a16 import cli


class CommandTests(unittest.TestCase):
    def test_help_and_cpu_commands_do_not_import_gpu_libraries(self):
        import subprocess
        import sys
        code = "import sys; from e38_w4a16.cli import parser; parser().format_help(); assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        subprocess.run([sys.executable, '-c', code], check=True)

    def test_generation_requires_explicit_permission_to_execute_the_pinned_base_code(self):
        options = cli.parser().parse_args(['generate', '--base-dir', 'base', '--overlay-dir', 'overlay', '--prompt', 'test'])
        with self.assertRaisesRegex(ValueError, 'trust-base-code'):
            cli.new_engine(options)

    def test_build_passes_only_explicitly_supplied_model_directories(self):
        result = {'complete': True}
        with patch('e38_w4a16.checkpoint.build_overlay', return_value=result) as build, redirect_stdout(io.StringIO()) as output:
            cli.main(['build', '--base-dir', 'base', '--source-dir', 'source', '--output', 'overlay'])
        build.assert_called_once_with(Path('base'), Path('source'), Path('overlay'))
        self.assertEqual(json.loads(output.getvalue()), result)
