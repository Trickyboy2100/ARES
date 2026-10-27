# 文档构建与视觉验收

2026-10-27。最终中文方案9页、英文两栏稿10页、导师简报2页；21页均渲染为PNG并通过版面目检。最后一次可读性修复仅把英文Fig4历史曲线、Fig7本轮曲线改为跨双栏，未重绘源图、未改物理或数值；中文两份PDF字节/哈希保持不变。英文10页重新渲染，特别放大检查第7/10页曲线轴、图例及图注。检查包括页眉/页脚、标题层级、中文/英文/希腊符号、表格边界、公式、图注、截图来源说明及跨页段落；无可见重叠、裁切、黑方块或缺字。英文参考文献采用8.5pt，正文10pt，所有宽幅方法/结果/学习图跨双栏以保证打印可读性。

中文采用可编辑Markdown + `build_client_documents.py`（ReportLab）；Droid CJK和DejaVu字形按字符显式选择，构建时先检查所有MD字符是否有支持的字形，防止中文字体缺失Latin/Greek而静默留白。实际内容字体均嵌入PDF。英文使用系统pdfTeX/BibTeX编译完整多文件项目，不是LaTeX截图或仅有模板。无需系统安装、驱动升级或外部大型依赖。

执行顺序：

```bash
bash 02_PAPER_SEED/build.sh
python build_client_documents.py
python verify_client_documents.py
/usr/bin/pdftoppm -scale-to 950 -png 01_CLIENT_PROPOSAL_CN.pdf /tmp/ares_client_pdf_qa/release_proposal
/usr/bin/pdftoppm -scale-to 950 -png 02_PAPER_SEED/main.pdf /tmp/ares_client_pdf_qa/release_paper
/usr/bin/pdftoppm -scale-to 950 -png 07_MENTOR_BRIEF_CN.pdf /tmp/ares_client_pdf_qa/release_brief
```

仅重编英文时使用 `python build_client_documents.py --manifest-only` 刷新3份输出哈希，不重新生成已验收的中文PDF。最终英文可读性修复的渲染目录为`/tmp/ares_client_pdf_qa/readability_paper-*.png`。

本机Python为已存在的Codex bundled runtime；英文编译为系统TeX Live 2022/dev/Debian。bundled Poppler因GLIBC要求不兼容，使用已存在的`/usr/bin/pdftoppm`回退，未安装系统软件。PNG目检中间文件位于上述明确`/tmp`目录，不作为实验原始证据。

自动检查结果为PASS：3份PDF字节/页数/文本；无replacement/null字符；196,608本轮预算被保留；英文8节与10条引用齐全；最终日志无Warning、undefined或Overfull；13份论文内图/表与当前权威生成输入逐字节SHA一致。JSON记录精确PDF及输入哈希，避免用后续重绘图替换已编译稿而不更新来源。

结果口径核对：历史v6单列P1/P2，不称P2已解决；新P2 B0=30，B1=30/29/30，B2/B3=30/30/30（各32例）。同模型输入干预不是独立训练seed；成本表中的推理行引用基础checkpoint的训练预算，不重复计训练，物理评估并非零开销。新G3不通过、最终测试未启封、三种子和短pilot局限明确。B1/B2并非已证明信息完全等价或全状态等变。截图为历史v6b策略LIVE重评，55/64独立通过，展示帧非终态；不是B3训练、固定回放、实机或现场料盘。

`DOCUMENT_BUILD_MANIFEST.json`与`DOCUMENT_QA.json`为当前可核验输出。文稿是甲方可读的研究初稿，不是已验证新算法或可投稿保证。
