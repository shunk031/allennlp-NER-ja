import os
import pathlib

from allennlp.common.testing.test_case import AllenNlpTestCase


class AllennlpNerJaTestCase(AllenNlpTestCase):
    PROJECT_ROOT = (pathlib.Path(__file__).parent / ".." / ".." / "..").resolve()
    MODULE_ROOT = PROJECT_ROOT / "allennlp_ner_ja"
    TOOLS_ROOT = MODULE_ROOT / "tools"
    PROJECT_ROOT_FALLBACK = (
        # users wanting to run test suite for installed package
        pathlib.Path(os.environ["ALLENNLP_SRC_DIR"])
        if "ALLENNLP_SRC_DIR" in os.environ
        else (
            # fallback for conda packaging
            pathlib.Path(os.environ["SRC_DIR"])
            if "CONDA_BUILD" in os.environ
            # stay in-tree
            else PROJECT_ROOT
        )
    )
    TESTS_ROOT = PROJECT_ROOT_FALLBACK / "tests"
    FIXTURES_ROOT = PROJECT_ROOT_FALLBACK / "test_fixtures"
