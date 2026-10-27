# 当前文档编译、渲染与证据核验

版本：2026-10-27 closed-loop continuation；不是旧v7校验记录。旧整包、原PDF/MD/LaTeX及SHA在非Git历史目录保全，见08/references/V7_DOCUMENT_PRESERVATION.json。

## 实际输出

- 中文方案10页：第2页更新摘要，第10页为本轮三次研究/独立结果/机器生成简表；第3–9页明确标HISTORICAL v6/v7，避免将旧正结果冒充新方法。
- 英文两栏LaTeX/PDF18页：题目/摘要、Related Work、问题、实际方法、实验、三轮真实结果、局限及参考文献齐全。当前11项参考，References集中末页，不穿插在末尾物理图之间。
- 导师简报2页；新增可视化短册4页。短册前三页为真实同案例filmstrip，第四页含R3单seed真实救援、完整三seed负结果及独立真实在线训练16相机同tick拼版。

四份当前PDF精确大小、SHA与页数见DOCUMENT_BUILD_MANIFEST.json和DOCUMENT_QA.json，不复用旧PUBLIC_FILE_HASHES。39项论文图/表副本按字节SHA验证。新正式TeX表来自已完成run索引，按checkpoint SHA绑定真实训练预算，排除相机重复/训练8case与未验收行。

## 实际编译和渲染

使用系统pdfTeX/BibTeX（TeX Live 2022/dev/Debian），英文最终编译无undefined reference、LaTeX Warning、Overfull box。中文由ReportLab和已嵌入的CJK/DejaVu字体生成；不安装系统级依赖。bundled Poppler存在GLIBC要求差异，改用已有/usr/bin/pdftoppm，未修改主机环境。

实际渲染和查看全部34页，QA PNG保存在非Git runs/client_sci_document_qa_20261027/final/。英文全18页以逐页图/六页contactsheet检查，修改后重新查看13–18页；中文10页以contactsheet及第2/10页原尺寸检查；导师2页及短册4页均查看。未见文字裁切、乱码方框、图片覆盖正文、页脚碰撞或多余空白页。

修正过的打印问题：新FigR_method从过宽4列改为7in紧凑2×2源，正文9.3pt，入稿约8.8pt；新学习/逐案例图改为7.2in打印源，宽图跨两栏。中文第10页不缩小多行案例图，改为直接读取R3/fixed配对JSON的可读简表。完整矩阵仍作为独立矢量图及英文整页图交付。原03历史图保留，不改其科学数据。

## 内容一致性与诚实边界

所有PDF均包含实际总交互414,720：正式393,216、诊断5,120、成功训练录制12,288、失败录制4,096。失败运行0优化/已知工作和未知总physics成本分别记录，不猜测。不能把66完成run、1174评价次数或三seed重复32案例当独立训练/场景样本。

R3主指标22/25/20对24/24/18、强反馈26，未支持稳定增益；固定保持同25成功、共同成功快0.208s的单seed次指标保留，不反向改主指标。首512真实同样本梯度差.405706只证明机制接入，不混入较大的诊断smoke梯度，不等同学会恢复。87D/5D是正式接口，旧84D仅smoke；entropy coefficient为0。

正式训练保存60Hz完整pre-action/action记录与episode post-step状态；普通run的120Hz全过程是evaluation验收。独立展示训练另有120Hz/44字段sidecar，不虚称87D全量120Hz训练日志。16相机montage与真正单相机wide视角明确区分；在线训练展示真实三次rollout/288次优化，不是评价影片改标题。旧v7 live评价仍单列历史。

本包不读取/公开现场点云、配置或派生现场图。几何/任务验收不是完整CCD、自碰撞或实机安全认证；现阶段不宣称SCI可投稿或已验证新算法优势。
