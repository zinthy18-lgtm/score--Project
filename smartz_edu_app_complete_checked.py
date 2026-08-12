import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture

st.set_page_config(page_title='SmartZ-EDU', page_icon='🎓', layout='wide')

st.markdown('''<style>
.stApp { background:#f7f9fc; }
.block-container { max-width:1450px; padding-top:1.1rem; padding-bottom:2rem; }
.hero {
    background:linear-gradient(135deg,#173b72 0%,#2563a8 55%,#4b9bc4 100%);
    color:white; padding:1.7rem 2rem; border-radius:20px;
    margin-bottom:1.15rem; box-shadow:0 8px 24px rgba(23,59,114,.16);
}
.hero h1 { margin:0; color:white; font-size:2.05rem; font-weight:800; letter-spacing:-.02em; }
.hero p { margin:.4rem 0 .8rem; color:rgba(255,255,255,.92); font-size:1rem; }
.hero .badge { display:inline-block; padding:.28rem .7rem; margin:.15rem .25rem 0 0;
    border:1px solid rgba(255,255,255,.3); border-radius:999px;
    background:rgba(255,255,255,.13); font-size:.78rem; }
.section-title { font-weight:800; color:#173b72; margin-top:.4rem; }
.info-card,.result-card { background:white; border:1px solid #e5eaf1;
    border-radius:15px; padding:1rem 1.1rem; box-shadow:0 3px 12px rgba(15,23,42,.045); }
.ok { background:#ecfdf5; border-left:5px solid #16a34a; padding:.8rem 1rem; border-radius:10px; }
.warn { background:#fff7ed; border-left:5px solid #f59e0b; padding:.8rem 1rem; border-radius:10px; }
.bad { background:#fef2f2; border-left:5px solid #dc2626; padding:.8rem 1rem; border-radius:10px; }
.small { color:#64748b; font-size:.86rem; }
div[data-testid="stMetric"] { background:white; border:1px solid #e5eaf1;
    border-radius:14px; padding:.65rem .8rem; box-shadow:0 3px 10px rgba(15,23,42,.04); }
div[data-testid="stMetricLabel"] { color:#64748b; }
.stTabs [data-baseweb="tab-list"] { gap:.35rem; }
.stTabs [data-baseweb="tab"] { border-radius:10px 10px 0 0; padding:.55rem .9rem; }
</style>''', unsafe_allow_html=True)

st.markdown('''<div class="hero"><h1>🎓 SmartZ-EDU</h1><p>Hệ thống Z-score thích ứng bằng Mô hình Hỗn hợp Gauss Mềm (Soft GMM)</p><span>🧠 Soft GMM &nbsp; • &nbsp; 🔄 Adaptive BIC &nbsp; • &nbsp; 🚩 Ngoại lệ sư phạm &nbsp; • &nbsp; 📈 Theo dõi tiến bộ</span></div>''', unsafe_allow_html=True)

with st.expander("📌 Tổng quan nghiên cứu", expanded=True):
    st.markdown("""
    <div class="info-card">
    <b>🎯 Mục tiêu:</b> hỗ trợ đánh giá học sinh công bằng hơn khi phổ điểm có thể gồm nhiều cụm năng lực,
    bằng cách kết hợp Z-score với mô hình hỗn hợp Gauss mềm (Soft GMM).<br><br>
    <b>🔬 Quy trình:</b> làm sạch dữ liệu → kiểm tra phân phối → chọn số cụm bằng BIC →
    tính Z-score truyền thống và Soft GMM → phát hiện ngoại lệ → theo dõi tiến bộ.<br><br>
    <b>💡 Ý nghĩa:</b> hỗ trợ phân tích dữ liệu giáo dục và tham khảo trong công tác bồi dưỡng,
    phụ đạo; không thay thế quyết định chuyên môn của nhà trường.
    </div>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.header('📂 Dữ liệu')
    uploaded=st.file_uploader('Chọn Excel/CSV',type=['xlsx','xls','csv'])
    st.divider(); st.header('⚙️ Phân tích')
    k_max=st.slider('Số đỉnh tối đa',2,6,4)
    threshold=st.slider('Ngưỡng γ ngoại lệ',0.5,0.9,0.7,0.05)
    st.caption('Hệ thống tự dò k=1..k_max bằng BIC.')

if uploaded is None:
    st.info('👈 Tải file điểm số ở thanh bên trái để bắt đầu.'); st.stop()
try:
    raw0=pd.read_csv(uploaded,header=None) if uploaded.name.lower().endswith('.csv') else pd.read_excel(uploaded,header=None)
except Exception as e:
    st.error(f'Không đọc được file: {e}'); st.stop()

preview=min(10,len(raw0)); header_guess=int(raw0.head(preview).notna().sum(axis=1).idxmax())
with st.expander('📄 Xem trước & chọn dòng tiêu đề'):
    st.dataframe(raw0.head(preview),use_container_width=True)
    header=st.number_input('Dòng tiêu đề (đánh số từ 0)',0,max(0,preview-1),header_guess)
raw=raw0.iloc[header+1:].copy(); raw.columns=raw0.iloc[header].values
raw=raw.dropna(axis=1,how='all'); raw=raw.loc[:,[c for c in raw.columns if pd.notna(c)]].copy(); raw.index=range(len(raw))
cols=raw.columns.tolist()

st.markdown('### 🎯 Thiết lập dữ liệu phân tích')
c1,c2=st.columns(2)
with c1: group_col=st.selectbox('Cột loại hình lớp',cols)
with c2: mode=st.radio('Chế độ',['Một cột điểm','Nhiều cột điểm'],horizontal=True)
score_candidates=[c for c in cols if c!=group_col]
if not score_candidates: st.error('Không có cột điểm để phân tích.'); st.stop()
if mode=='Một cột điểm': score_cols=[st.selectbox('Cột điểm',score_candidates)]
else: score_cols=st.multiselect('Các cột điểm',score_candidates,default=score_candidates[:2])
if not score_cols: st.warning('Chọn ít nhất một cột điểm.'); st.stop()

def fit_one(x,k):
    g=GaussianMixture(n_components=k,n_init=20,random_state=42).fit(x.reshape(-1,1))
    means=g.means_.ravel(); stds=np.sqrt(g.covariances_).ravel(); weights=g.weights_.ravel(); order=np.argsort(means)
    return [(means[i],max(stds[i],1e-8),weights[i]) for i in order],g.bic(x.reshape(-1,1))

def analyze(col):
    d=raw[[group_col,col]].copy(); d[col]=pd.to_numeric(d[col],errors='coerce'); d=d.dropna()
    if len(d)<5:return None
    x=d[col].to_numpy(float); max_eff=max(1,min(k_max,len(x)//5)); bic={}; comps={}
    for k in range(1,max_eff+1): comps[k],bic[k]=fit_one(x,k)
    best=min(bic,key=bic.get); res=d.copy(); res['Z_truyen_thong']=((x-x.mean())/x.std(ddof=1)).round(3); gam=None
    if best>=2:
        dens=np.array([pi*stats.norm.pdf(x,mu,s) for mu,s,pi in comps[best]]); gam=dens/np.maximum(dens.sum(axis=0),1e-300)
        ze=np.array([(x-mu)/s for mu,s,_ in comps[best]]); res['Z_GMM_mềm']=(gam*ze).sum(axis=0).round(3); res['Z_GMM_cứng']=ze[np.argmax(gam,axis=0),np.arange(len(x))].round(3)
        for i in range(best): res[f'gamma_cụm_{i+1}']=gam[i].round(4)
    return {'data':d,'x':x,'bic':bic,'best':best,'components':comps[best],'result':res,'gam':gam}

results={c:analyze(c) for c in score_cols}; results={k:v for k,v in results.items() if v is not None}
if not results: st.error('Không đủ dữ liệu hợp lệ.'); st.stop()

for score_col,r in results.items():
    x=r['x']; best=r['best']; res=r['result']; st.markdown(f'## 📊 {score_col}')
    c=st.columns(6); vals=[len(x),res[group_col].nunique(),x.mean(),x.std(ddof=1),best,r['bic'][best]]; labs=['👨‍🎓 Học sinh','🏫 Nhóm lớp','📈 Điểm TB','σ','🧠 k tối ưu','BIC']
    for i in range(6): c[i].metric(labs[i],f'{vals[i]:.2f}' if isinstance(vals[i],float) else vals[i])
    sh_w,sh_p=stats.shapiro(x) if len(x)<=5000 else (np.nan,np.nan)
    with st.expander('🔬 Thống kê mô tả & kiểm định tính chuẩn'):
        a,b=st.columns(2)
        with a: st.dataframe(res.groupby(group_col)[score_col].agg(['count','mean','std','median']).round(3),use_container_width=True)
        with b:
            st.metric('Shapiro–Wilk W',f'{sh_w:.4f}'); st.metric('p-value',f'{sh_p:.4g}')
            st.error('p < 0.05 → dữ liệu không phù hợp với phân phối chuẩn.') if sh_p<0.05 else st.success('p ≥ 0.05 → chưa có bằng chứng bác bỏ phân phối chuẩn.')
    st.success('Hệ thống tự chọn k=2 bằng BIC → áp dụng Soft GMM.') if best==2 else st.info(f'Hệ thống tự chọn k={best} bằng BIC → '+('Z-score truyền thống.' if best==1 else 'Soft GMM tổng quát.'))
    tabs=st.tabs(['📈 Phổ điểm','🧠 GMM & BIC','⚖️ Z-score','🚩 Ngoại lệ','📋 Kết quả'])
    with tabs[0]:
        fig,ax=plt.subplots(figsize=(10,4.8)); lo,hi=float(np.floor(x.min())),float(np.ceil(x.max())); bins=np.arange(max(0,lo-.5),hi+1,.5)
        ax.hist(x,bins=bins,density=True,alpha=.78,edgecolor='white',label='Phổ điểm'); xs=np.linspace(bins[0],bins[-1],400); total=np.zeros_like(xs)
        for i,(mu,s,pi) in enumerate(r['components']):
            curve=pi*stats.norm.pdf(xs,mu,s); total+=curve
            if best>=2: ax.plot(xs,curve,'--',lw=2,label=f'Cụm {i+1}: μ={mu:.2f}, w={pi:.2f}')
        ax.plot(xs,total if best>=2 else stats.norm.pdf(xs,x.mean(),x.std(ddof=1)),lw=2.5,label=f'GMM ({best} cụm)' if best>=2 else 'Chuẩn')
        ax.set_xlabel('Điểm'); ax.set_ylabel('Mật độ'); ax.grid(alpha=.2,ls='--'); ax.legend(fontsize=8); st.pyplot(fig,use_container_width=True); plt.close(fig)
    with tabs[1]:
        a,b=st.columns([1.3,1])
        with a:
            fig,ax=plt.subplots(figsize=(6,3.6)); ks=list(r['bic']); ax.bar([str(k) for k in ks],[r['bic'][k] for k in ks]); ax.set_xlabel('k'); ax.set_ylabel('BIC'); ax.set_title('Chọn số cụm bằng BIC'); ax.grid(axis='y',alpha=.2); st.pyplot(fig,use_container_width=True); plt.close(fig)
        with b: st.dataframe(pd.DataFrame({'k':ks,'BIC':np.round([r['bic'][k] for k in ks],2),'Được chọn':[k==best for k in ks]}),use_container_width=True,hide_index=True)
        if best>=2: st.dataframe(pd.DataFrame(r['components'],columns=['Mean','Std','Weight'],index=range(1,best+1)).round(3),use_container_width=True)
    with tabs[2]:
        if best>=2:
            zc=res[['Z_truyen_thong','Z_GMM_mềm','Z_GMM_cứng']].copy(); zc['Chênh lệch mềm - truyền thống']=(zc['Z_GMM_mềm']-zc['Z_truyen_thong']).round(3); st.dataframe(zc,use_container_width=True,height=350); st.caption('Z GMM mềm dùng xác suất γ để tránh ép học sinh ở vùng ranh giới vào một cụm duy nhất.')
        else: st.info('k=1 nên chưa cần Z-score GMM.')
    with tabs[3]:
        if best==2 and r['gam'] is not None and res[group_col].nunique()==2:
            means=res.groupby(group_col)[score_col].mean(); low,high=means.idxmin(),means.idxmax(); up=res[(res[group_col]==low)&(res['gamma_cụm_2']>threshold)]; down=res[(res[group_col]==high)&(res['gamma_cụm_2']<1-threshold)]
            a,b=st.columns(2)
            with a: st.markdown(f'<div class="ok"><b>{len(up)} HS</b> nhóm {low} có γ cụm cao &gt; {threshold:.2f}</div>',unsafe_allow_html=True); st.dataframe(up,use_container_width=True,height=250)
            with b: st.markdown(f'<div class="bad"><b>{len(down)} HS</b> nhóm {high} có γ cụm thấp &lt; {1-threshold:.2f}</div>',unsafe_allow_html=True); st.dataframe(down,use_container_width=True,height=250)
            st.caption('γ là xác suất hậu nghiệm: càng gần 1, học sinh càng có xu hướng thuộc cụm đó.')
        else: st.info('Phát hiện ngoại lệ sư phạm được áp dụng khi k=2 và có đúng 2 nhóm.')
    with tabs[4]:
        st.dataframe(res,use_container_width=True,height=420); buf=io.BytesIO()
        with pd.ExcelWriter(buf,engine='openpyxl') as w: res.to_excel(w,index=False,sheet_name='KetQua')
        st.download_button('📥 Tải kết quả Excel',buf.getvalue(),f'SmartZ_{score_col}.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',key=f'dl_{score_col}')

if len(results)>=2:
    st.markdown('---'); st.header('📈 Theo dõi tiến bộ GK → CK'); keys=list(results); a,b=st.columns(2)
    with a: cf=st.selectbox('Từ cột',keys,key='from')
    with b: ct=st.selectbox('Sang cột',keys,index=min(1,len(keys)-1),key='to')
    if cf!=ct and results[cf]['best']>=2 and results[ct]['best']>=2:
        m=pd.concat([raw[[group_col]],results[cf]['result']['Z_GMM_mềm'].rename('Z_from'),results[ct]['result']['Z_GMM_mềm'].rename('Z_to')],axis=1).dropna(); m['Tiến bộ']=m.Z_to-m.Z_from; st.dataframe(m.groupby(group_col)['Tiến bộ'].agg(['mean','std','count']).round(3),use_container_width=True)
    else: st.info('Cần hai cột điểm và cả hai cột phải có k≥2 để so sánh Z GMM mềm.')

st.markdown('---'); st.caption('SmartZ-EDU • Adaptive Soft Gaussian Mixture Model • Hỗ trợ phân tích giáo dục, không thay thế quyết định chuyên môn của nhà trường.')
