# -*- coding: utf-8 -*-
"""
Feed-Agent Web 界面（零依赖，仅用 Python 标准库 http.server）
用法：python app.py  →  浏览器打开 http://127.0.0.1:8611
页面：
  /      聊天界面（工具调用轨迹、SQL、溯源状态可展开）
  /eval  评测结果总览（读取 eval/CHANGELOG 数据与 results_v*.json）
接口：
  POST /api/chat   {"question": "..."} → {answer, trace, provenance, ...}
  GET  /api/eval   → 各版本评测汇总
"""
import os, sys, json, glob
from http.server import HTTPServer, BaseHTTPRequestHandler
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "agent"))
import agent

CFG = yaml.safe_load(open(os.path.join(HERE, "config.yaml"), encoding="utf-8"))
# 端口/主机优先级：命令行参数（argparse，见文件末尾）> 环境变量 > config.yaml > 默认 7100
HOST = os.environ.get("HOST") or CFG.get("web", {}).get("host", "127.0.0.1")
PORT = int(os.environ.get("PORT") or CFG.get("web", {}).get("port", 7100))

PAGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>饲料供应链经营分析 Agent</title>
<style>
:root{--bg:#f5f6f8;--card:#fff;--line:#e3e6ea;--pri:#1a6ee0;--ok:#178a4c;--bad:#c0392b;--mut:#8a919b}
*{box-sizing:border-box;margin:0}
body{font-family:"Microsoft YaHei","PingFang SC",sans-serif;background:var(--bg);color:#222;display:flex;flex-direction:column;height:100vh}
header{background:#14335c;color:#fff;padding:12px 20px;display:flex;align-items:baseline;gap:14px}
header h1{font-size:17px;font-weight:600}
header .sub{font-size:12px;opacity:.75}
header a{color:#9fc2f0;font-size:12px;margin-left:auto}
#chat{flex:1;overflow-y:auto;padding:18px 20px}
.msg{max-width:860px;margin:0 auto 14px}
.msg .bubble{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;white-space:pre-wrap;line-height:1.65;font-size:14px}
.msg.user .bubble{background:#e8f1fd;border-color:#c9defa}
.msg .who{font-size:12px;color:var(--mut);margin-bottom:4px}
.meta{display:flex;gap:8px;margin-top:8px;flex-wrap:wrap}
.tag{font-size:11px;padding:2px 8px;border-radius:10px;background:#eef1f4;color:#5a636e}
.tag.ok{background:#e3f5ea;color:var(--ok)}
.tag.bad{background:#fbe7e4;color:var(--bad)}
details{margin-top:8px;border-top:1px dashed var(--line);padding-top:6px}
summary{font-size:12px;color:var(--pri);cursor:pointer}
.trace{font-size:12px;color:#445;background:#f7f9fb;border-radius:6px;padding:8px;margin-top:6px;white-space:pre-wrap;word-break:break-all}
#chips{max-width:860px;margin:0 auto 10px;display:flex;gap:8px;flex-wrap:wrap}
.chip{font-size:12px;background:#fff;border:1px solid var(--line);border-radius:14px;padding:5px 12px;cursor:pointer;color:#3a62a0}
.chip:hover{border-color:var(--pri)}
#bar{padding:12px 20px;background:#fff;border-top:1px solid var(--line)}
#bar .row{max-width:860px;margin:0 auto;display:flex;gap:10px}
#q{flex:1;border:1px solid var(--line);border-radius:8px;padding:10px 12px;font-size:14px;font-family:inherit}
button{background:var(--pri);color:#fff;border:0;border-radius:8px;padding:0 22px;font-size:14px;cursor:pointer}
button:disabled{opacity:.5}
</style></head><body>
<header><h1>饲料供应链经营分析 Agent</h1>
<span class="sub" id="mode"></span>
<a href="/eval">评测结果</a></header>
<div id="chat"></div>
<div id="bar"><div id="chips"></div>
<div class="row"><input id="q" placeholder="例如：8月成都厂草鱼料的吨原料成本为什么比7月涨了？" autofocus>
<button id="send" onclick="send()">发送</button></div></div>
<script>
const CHIPS=["8月成都厂草鱼料的吨原料成本为什么比7月涨了？","如果豆粕下个月再涨10%，草鱼料配方该怎么调？",
"库存周转天数是多少","2025Q3哪个供应商的豆粕质量有问题？","哪个区域的回款变差了？什么时候开始的？","公司毛利率是多少？"];
const chat=document.getElementById('chat'),q=document.getElementById('q');
const chips=document.getElementById('chips');
CHIPS.forEach(c=>{const d=document.createElement('span');d.className='chip';d.textContent=c;
d.onclick=()=>{q.value=c;send()};chips.appendChild(d)});
fetch('/api/meta').then(r=>r.json()).then(m=>{document.getElementById('mode').textContent=
'模式：'+m.mode+(m.mode==='offline'?'（规则路由，配置API可切换LLM）':'（'+m.model+'）')+' ｜ 数据：'+m.db_period});
function addMsg(who,html){const d=document.createElement('div');d.className='msg '+(who==='我'?'user':'bot');
d.innerHTML='<div class="who">'+who+'</div><div class="bubble">'+html+'</div>';
chat.appendChild(d);chat.scrollTop=chat.scrollHeight;return d}
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
async function send(){
 const text=q.value.trim();if(!text)return;q.value='';
 addMsg('我',esc(text));
 const btn=document.getElementById('send');btn.disabled=true;
 const thinking=addMsg('Agent','思考中…');
 try{
  const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({question:text})});
  const d=await r.json();
  const prov=d.provenance;
  let html=esc(d.answer);
  html+='<div class="meta"><span class="tag '+(prov.passed?'ok':'bad')+'">溯源'+(prov.passed?'✓ 全部可溯':'✗ '+prov.unmatched.join(',')+' 未匹配')+'</span>'
   +'<span class="tag">工具调用 '+d.n_tool_calls+' 次</span><span class="tag">'+d.elapsed_s+'s</span>'
   +'<span class="tag">'+d.mode+'</span></div>';
  if(d.trace&&d.trace.length){
   html+='<details><summary>工具调用轨迹 / SQL</summary><div class="trace">'+esc(JSON.stringify(d.trace,null,1))+'</div></details>'}
  thinking.querySelector('.bubble').innerHTML=html;
 }catch(e){thinking.querySelector('.bubble').textContent='请求失败：'+e}
 btn.disabled=false;q.focus();chat.scrollTop=chat.scrollHeight}
q.addEventListener('keydown',e=>{if(e.key==='Enter')send()});
</script></body></html>"""

EVAL_PAGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>评测结果 - Feed-Agent</title>
<style>
body{font-family:"Microsoft YaHei",sans-serif;background:#f5f6f8;margin:0;padding:24px}
main{max-width:900px;margin:0 auto;background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:24px}
h1{font-size:18px;margin-bottom:6px}
a{color:#1a6ee0;font-size:13px}
table{border-collapse:collapse;width:100%;margin-top:16px;font-size:13px}
th,td{border:1px solid #e3e6ea;padding:8px 10px;text-align:center}
th{background:#f0f4f9}
.note{font-size:12px;color:#8a919b;margin-top:14px;line-height:1.7}
</style></head><body><main>
<h1>离线评测结果（30 题）</h1><a href="/">← 返回聊天</a>
<table id="t"><tr><th>版本</th><th>准确率</th><th>数字溯源通过率</th><th>平均工具调用</th><th>平均耗时</th><th>失败题</th></tr></table>
<div class="note">评测集：eval/questions.yaml（单指标10/下钻环比8/归因5/what-if4/澄清1/拒答2），
判分脚本 eval/run_eval.py，逐题回答原文留档于 eval/results_v*.json。<br>
基线A（LLM裸写SQL、无语义层护栏）需在 config.yaml 配置 API key 后运行，当前留空不编造。<br>
offline 模式为确定性规则路由，100% 说明题库被规则覆盖，不代表泛化能力；真实泛化以 llm 模式评测为准。</div>
</main><script>
fetch('/api/eval').then(r=>r.json()).then(rows=>{
const t=document.getElementById('t');
rows.forEach(r=>{t.innerHTML+='<tr><td>'+r.version+'</td><td>'+(r.accuracy*100).toFixed(1)+'%</td><td>'
+(r.provenance_pass_rate*100).toFixed(0)+'%</td><td>'+r.avg_tool_calls+'</td><td>'+r.avg_elapsed_s+'s</td>'
+'<td>'+(r.failed_ids.join('、')||'无')+'</td></tr>'})});
</script></body></html>"""

class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype="text/html; charset=utf-8", code=200):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self._send(PAGE)
        elif self.path == "/eval":
            self._send(EVAL_PAGE)
        elif self.path == "/api/meta":
            import duckdb
            db = os.path.join(HERE, CFG["paths"]["db"])
            con = duckdb.connect(db, read_only=True)
            period = con.execute("SELECT MIN(month)||' ~ '||MAX(month) FROM fact_sales").fetchone()[0]
            con.close()
            mode = CFG.get("mode", "offline")
            self._send(json.dumps({"mode": mode, "model": CFG.get("llm", {}).get("model", ""),
                                   "db_period": period}, ensure_ascii=False),
                       "application/json; charset=utf-8")
        elif self.path == "/api/eval":
            rows = []
            for fp in sorted(glob.glob(os.path.join(HERE, "eval", "results_v*.json"))):
                rows.append(json.load(open(fp, encoding="utf-8"))["summary"])
            self._send(json.dumps(rows, ensure_ascii=False), "application/json; charset=utf-8")
        else:
            self._send("404", code=404)

    def do_POST(self):
        if self.path == "/api/chat":
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            question = body.get("question", "").strip()
            if not question:
                self._send(json.dumps({"error": "empty question"}), "application/json", 400)
                return
            result = agent.ask(question, mode=CFG.get("mode", "offline"))
            self._send(json.dumps(result, ensure_ascii=False, default=str),
                       "application/json; charset=utf-8")
        else:
            self._send("404", code=404)

    def log_message(self, fmt, *args):
        pass  # 静默访问日志

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    print(f"Feed-Agent 已启动: http://{args.host}:{args.port}  （Ctrl+C 停止）")
    HTTPServer((args.host, args.port), Handler).serve_forever()
