from pathlib import Path
import tempfile
import unittest
from cctv_ai.growth_serve import resolve_asset


class GrowthServeTests(unittest.TestCase):
    def test_catalog_is_explicit_and_blocks_windows_and_url_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);root=base/'first';other=base/'second';root.mkdir();other.mkdir()
            (root/'index.html').write_text('first')
            (other/'index.html').write_text('second')
            (base/'private.txt').write_text('outside')
            catalog={'second':str(other)}
            self.assertEqual(resolve_asset(root,catalog,'/index.html'),root/'index.html')
            self.assertEqual(resolve_asset(root,catalog,'/batches/second/index.html'),other/'index.html')
            for path in ['/../private.txt','/%2e%2e/private.txt','/batches/unknown/index.html',
                         '/batches/second/../../private.txt','/%5c%5cother-host%5cshare%5cfile','/C:/private.txt']:
                self.assertIsNone(resolve_asset(root,catalog,path),path)


if __name__=='__main__':unittest.main()
