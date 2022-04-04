from __future__ import annotations

import git


def get_hashes() -> dict[str, str]:
    repos = {"eddydiff": "~/work/eddydiff/", "dcpy": "~/python/dcpy/"}
    hashes = {}
    for name, dirname in repos.items():
        repo = git.Repo(dirname)
        hashes[name] = repo.head.commit.hexsha
        repo.close()
    return hashes
