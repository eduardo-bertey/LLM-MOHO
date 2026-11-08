"""Especializacion de expertos por CONTRIBUCION, no por gating.

Idea (SAE Interprets MoE, arXiv:2611.08089): que el router elija a un
experto no prueba que ese experto haga el trabajo. La salida de la MoE
es suma ponderada de las salidas de los expertos, asi que la
contribucion de cada experto se puede medir exacta, sin SAE:
    contrib_e = w_e * FF_e(x)   (w_e = peso router renormalizado top-k)
y_total = sum_e contrib_e.

Mide, sobre el eval v7 (2361 vectores, nunca entrenados):
- reproduccion de la eval final (cos/R2/NMSE) para verificar el checkpoint
- seleccion vs contribucion: fraccion de tokens donde cada experto es
  seleccionado, y fraccion donde es el mayor contribuyente
- cuota de norma: que parte de ||y|| aporta cada experto en promedio
- numero efectivo de expertos por token (por cuota de norma)
- si los expertos hacen lo mismo: coseno entre sus vectores medios de
  contribucion, y coseno medio entre salidas de pares de expertos en los
  tokens donde ambos estan seleccionados
- solapamiento de dominancia entre pares

Checkpoint: checkpoints/moe_d2dmoe_e3d2d_m4k3e5_resume.npz
  5 expertos, top-3, clamp -4, cos 0.9598 / R2 0.9120 / NMSE 0.0880
Datos: activaciones-reales-qwen25-v7.npz (x_eval, y_eval)
Solo numpy, lotes de 256 para no pedir RAM (VM ~550 Mi disponibles).
"""
import numpy as np, time
BASE = "/home/hatch/workspace/qwen-ff-local"
NEXP, TOPK, HIDDEN = 5, 3, 896

def sigmoid(z): return 1.0/(1.0+np.exp(-np.clip(z, -30, 30)))
def silu(z): return z*sigmoid(z)
def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e/np.sum(e, axis=1, keepdims=True)
def metrics(y, yt):
    # mismo cos que d2dmoe_resume_e3.py: media del coseno por vector
    cos = float(np.mean(np.sum(y*yt, axis=1)/(np.linalg.norm(y, axis=1)*np.linalg.norm(yt, axis=1)+1e-12)))
    ss_res = np.sum((yt-y)**2); ss_tot = np.sum((yt-yt.mean())**2)
    r2 = 1-ss_res/ss_tot
    nmse = ss_res/np.sum(yt**2)
    return cos, r2, nmse

t0 = time.time()
d = np.load(f"{BASE}/activaciones-reales-qwen25-v7.npz")
x_ev, y_ev = d["x_eval"].astype(np.float32), d["y_eval"].astype(np.float32)
print(f"eval v7: {x_ev.shape} vectores", flush=True)
c = np.load(f"{BASE}/checkpoints/moe_d2dmoe_e3d2d_m4k3e5_resume.npz")
exps = [ (c[f"gate_{e}"], c[f"up_{e}"], c[f"down_{e}"]) for e in range(NEXP) ]
Wr, br = c["router_W"], c["router_b"]
print("checkpoint cargado: 5 expertos, router", Wr.shape, flush=True)

n = len(x_ev)
sel = np.zeros(NEXP); dom = np.zeros(NEXP)
norm_share = np.zeros(NEXP)          # cuota media de norma (sobre suma de normas)
energy_share = np.zeros(NEXP)        # cuota media de <contrib, y>/||y||^2
mean_contrib = np.zeros((NEXP, HIDDEN), dtype=np.float64)
mean_contrib_sel = np.zeros((NEXP, HIDDEN), dtype=np.float64)
eff_n, top_share = [], []
pair_cos_sum = np.zeros((NEXP, NEXP)); pair_cos_cnt = np.zeros((NEXP, NEXP))
pair_dom = np.zeros((NEXP, NEXP))    # veces que i e j son los 2 mayores contribuyentes
y_all = np.zeros_like(y_ev)
B = 256
for i in range(0, n, B):
    xb = x_ev[i:i+B]
    logits = xb@Wr.T + br; probs = softmax(logits)
    idx = np.argpartition(-probs, TOPK, axis=1)[:, :TOPK]
    w = np.zeros_like(probs)
    np.put_along_axis(w, idx, np.take_along_axis(probs, idx, axis=1), axis=1)
    w = w/np.sum(w, axis=1, keepdims=True)
    y = np.zeros((len(xb), HIDDEN), dtype=np.float32)
    contribs = []
    for e in range(NEXP):
        wg, wu, wd = exps[e]
        h = silu(xb@wg.T)*(xb@wu.T)
        ye = h@wd.T
        ce = ye*w[:, e][:, None]
        contribs.append(ce)
        y += ce
    y_all[i:i+B] = y
    cn = np.stack([np.linalg.norm(ce, axis=1) for ce in contribs], axis=1)  # (b,5)
    sel += (w > 0).sum(axis=0)
    dom += (np.argmax(cn, axis=1)[:, None] == np.arange(NEXP)[None, :]).sum(axis=0)
    s = cn.sum(axis=1, keepdims=True); s[s == 0] = 1
    norm_share += (cn/s).sum(axis=0)
    yn = np.linalg.norm(y, axis=1, keepdims=True); yn[yn == 0] = 1
    for e in range(NEXP):
        energy_share[e] += np.sum(np.sum(contribs[e]*y, axis=1)/(yn[:, 0]**2))
        mean_contrib[e] += contribs[e].sum(axis=0)
        m = w[:, e] > 0
        mean_contrib_sel[e] += contribs[e][m].sum(axis=0) if m.any() else 0
    eff_n.extend((1.0/np.sum((cn/s)**2, axis=1)).tolist())
    top_share.extend((cn.max(axis=1)/s[:, 0]).tolist())
    # coseno entre salidas de pares de expertos (sin peso router, ye crudo)
    # lo calculamos sobre contribuciones: direccion de cada experto
    for a in range(NEXP):
        for b in range(a+1, NEXP):
            m = (w[:, a] > 0) & (w[:, b] > 0)
            if m.any():
                na = np.linalg.norm(contribs[a][m], axis=1); nb = np.linalg.norm(contribs[b][m], axis=1)
                ok = (na > 0) & (nb > 0)
                if ok.any():
                    cs = np.sum(contribs[a][m][ok]*contribs[b][m][ok], axis=1)/(na[ok]*nb[ok])
                    pair_cos_sum[a, b] += cs.sum(); pair_cos_cnt[a, b] += ok.sum()
    # pares dominantes: los 2 mayores contribuyentes por token
    order = np.argsort(-cn, axis=1)
    for t in range(len(xb)):
        a, b = sorted(order[t, :2].tolist())
        pair_dom[a, b] += 1
    if (i//B) % 4 == 0:
        print(f"  lote {i}/{n} ({time.time()-t0:.0f}s)", flush=True)

cos, r2, nmse = metrics(y_all, y_ev)
print(f"\n[REPRODUCCION EVAL] cos={cos:.4f} R2={r2:.4f} NMSE={nmse:.4f}  (esperado 0.9598/0.9120/0.0880)", flush=True)
print(f"tiempo {time.time()-t0:.0f}s", flush=True)
print("\n== POR EXPERTO (eval, n=%d) ==" % n)
print("exp | sel% | dom% | cuota_norma% | cuota_energia% | |contrib media|")
for e in range(NEXP):
    mc = mean_contrib[e]/n
    print(f" {e}  | {100*sel[e]/n:5.1f} | {100*dom[e]/n:5.1f} | {100*norm_share[e]/n:6.1f} | {100*energy_share[e]/n:7.1f} | {np.linalg.norm(mc):.4f}", flush=True)
eff_n = np.array(eff_n); top_share = np.array(top_share)
print(f"\n== POR TOKEN ==")
print(f"numero efectivo de expertos (por cuota de norma): media {eff_n.mean():.2f} | mediana {np.median(eff_n):.2f} | p10 {np.percentile(eff_n,10):.2f} p90 {np.percentile(eff_n,90):.2f}  (top-3 = max 3.0)")
print(f"cuota del mayor contribuyente: media {top_share.mean():.3f} | mediana {np.median(top_share):.3f} | tokens con >80% en un solo experto: {100*np.mean(top_share>0.8):.1f}% | tokens con <50%: {100*np.mean(top_share<0.5):.1f}%")
print("\n== COSENO ENTRE VECTORES MEDIOS DE CONTRIBUCION (1.0 = hacen lo mismo) ==")
mc = mean_contrib/n
for a in range(NEXP):
    row = []
    for b in range(NEXP):
        na, nb = np.linalg.norm(mc[a]), np.linalg.norm(mc[b])
        row.append(f"{np.dot(mc[a],mc[b])/(na*nb):+.3f}" if na*nb > 0 else "  -  ")
    print("  ".join(row), flush=True)
print("\n== COSENO MEDIO ENTRE SALIDAS DE PARES (tokens donde ambos seleccionados) ==")
for a in range(NEXP):
    for b in range(a+1, NEXP):
        if pair_cos_cnt[a, b]:
            print(f"  exp {a}-{b}: cos medio {pair_cos_sum[a,b]/pair_cos_cnt[a,b]:+.3f}  (n={int(pair_cos_cnt[a,b])})", flush=True)
print("\n== PARES QUE DOMINAN JUNTOS (top-2 contribuyentes, % tokens) ==")
for a in range(NEXP):
    for b in range(a+1, NEXP):
        if pair_dom[a, b]:
            print(f"  exp {a}-{b}: {100*pair_dom[a,b]/n:.1f}%", flush=True)
print("\nLECTURA: dom% concentrado en 1-2 expertos + coseno entre medias alto "
      "(>0.8) = expertos copias, poca especializacion. dom% repartido + "
      "coseno bajo + numero efectivo cerca de 3 = cada experto aporta algo distinto.", flush=True)
