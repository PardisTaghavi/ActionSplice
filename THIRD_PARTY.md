# Third-party software and models

ActionSplice integrates with, but does not redistribute, these upstream
repositories or their model weights:

| Project | Repository | Pinned revision | Upstream license |
|---|---|---|---|
| minWM | [shengshu-ai/minWM](https://github.com/shengshu-ai/minWM) | `df522a26cd4409d3e3e8f269cc98eac069b5df47` | [Apache-2.0](https://github.com/shengshu-ai/minWM/blob/df522a26cd4409d3e3e8f269cc98eac069b5df47/LICENSE) |
| HY-WorldPlay | [Tencent-Hunyuan/HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) | `1588e1336e842b03b0a7860c654ebd7c46bb065e` | [Tencent HY-WorldPlay Community License Agreement](https://github.com/Tencent-Hunyuan/HY-WorldPlay/blob/1588e1336e842b03b0a7860c654ebd7c46bb065e/License.txt) |

The HY-WorldPlay license includes territory and distribution restrictions; read
its `License.txt` before using that backend. Users are responsible for complying
with each upstream code and model license. Corrector releases should link to the
required upstream weights instead of repackaging them.
