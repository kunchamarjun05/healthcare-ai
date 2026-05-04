"""
🏥 MedPredict AI — Multi-Disease Prediction System
Professional Healthcare ML Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from models import (
    load_heart_data, load_diabetes_data, load_cancer_data,
    train_and_evaluate, predict_single
)

st.set_page_config(page_title="MedPredict AI", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

# ── Professional Green Medical Theme ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

*{font-family:'Plus Jakarta Sans',sans-serif;}
:root{
    --bg:#f4f9f6;--card:#ffffff;--card-hover:#f0f7f3;
    --primary:#0d9668;--primary-dark:#047857;--primary-light:#d1fae5;--primary-glow:rgba(13,150,104,0.12);
    --text:#1e293b;--text-secondary:#475569;--muted:#94a3b8;
    --border:#e2e8f0;--border-green:#a7f3d0;
    --danger:#dc2626;--danger-bg:#fef2f2;--danger-border:#fecaca;
    --success:#059669;--success-bg:#ecfdf5;--success-border:#a7f3d0;
}

.stApp{background:var(--bg)!important;color:var(--text);}

/* Sidebar */
section[data-testid="stSidebar"]{background:#ffffff!important;border-right:1px solid var(--border)!important;box-shadow:2px 0 12px rgba(0,0,0,0.04);}
section[data-testid="stSidebar"] .stMarkdown h1{color:var(--primary-dark)!important;font-size:1.3rem!important;}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] .stMarkdown div,
section[data-testid="stSidebar"] .stMarkdown label,
section[data-testid="stSidebar"] .stRadio label{color:var(--text)!important;}
section[data-testid="stSidebar"] .stMarkdown b,
section[data-testid="stSidebar"] .stMarkdown strong{color:var(--primary-dark)!important;}
section[data-testid="stSidebar"] hr{border-color:var(--border)!important;}

/* Hero Banner */
.hero-banner{
    background:linear-gradient(135deg,#047857 0%,#059669 40%,#10b981 100%);
    border-radius:16px;padding:36px 40px;margin-bottom:28px;position:relative;overflow:hidden;
    box-shadow:0 8px 32px rgba(5,150,105,0.2);
}
.hero-banner::before{content:'';position:absolute;top:-40%;right:-10%;width:280px;height:280px;
background:radial-gradient(circle,rgba(255,255,255,0.12),transparent 70%);border-radius:50%;}
.hero-banner::after{content:'';position:absolute;bottom:-50%;left:10%;width:200px;height:200px;
background:radial-gradient(circle,rgba(255,255,255,0.08),transparent 70%);border-radius:50%;}
.hero-banner h1{font-size:2rem;font-weight:800;margin:0;color:#fff!important;position:relative;z-index:1;}
.hero-banner p{color:#d1fae5!important;font-size:1rem;margin-top:6px;position:relative;z-index:1;opacity:0.95;}

/* Stat Cards */
.stat-card{
    background:var(--card);border:1px solid var(--border);border-radius:14px;
    padding:22px;text-align:center;transition:all 0.25s ease;
    box-shadow:0 1px 4px rgba(0,0,0,0.04);
}
.stat-card:hover{transform:translateY(-3px);box-shadow:0 8px 24px var(--primary-glow);border-color:var(--border-green);}
.stat-icon{font-size:2rem;margin-bottom:8px;}
.stat-value{font-size:1.8rem;font-weight:800;color:var(--primary);}
.stat-label{font-size:0.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.8px;margin-top:4px;font-weight:600;}

/* Result Cards */
.result-card{border-radius:14px;padding:28px;text-align:center;margin:20px 0;animation:slideUp 0.4s ease;}
.result-safe{background:var(--success-bg);border:2px solid var(--success-border);}
.result-safe h2{color:var(--success)!important;font-size:1.5rem;margin:0 0 6px;}
.result-safe p{color:var(--text-secondary);font-size:0.95rem;}
.result-danger{background:var(--danger-bg);border:2px solid var(--danger-border);}
.result-danger h2{color:var(--danger)!important;font-size:1.5rem;margin:0 0 6px;}
.result-danger p{color:var(--text-secondary);font-size:0.95rem;}
@keyframes slideUp{from{opacity:0;transform:translateY(12px);}to{opacity:1;transform:translateY(0);}}

/* Info Box */
.info-box{background:#f0fdf4;border:1px solid #bbf7d0;border-left:4px solid var(--primary);
border-radius:0 10px 10px 0;padding:14px 18px;margin:12px 0;font-size:0.88rem;color:var(--text-secondary);}

/* Section Header */
.section-hdr{display:flex;align-items:center;gap:10px;margin:24px 0 16px;padding-bottom:10px;border-bottom:2px solid var(--border);}
.section-hdr h3{margin:0;font-size:1.15rem;font-weight:700;color:var(--primary-dark)!important;}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{gap:4px;background:var(--card);padding:4px;border-radius:12px;border:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{border-radius:10px;color:var(--text-secondary);padding:8px 20px;font-weight:600;border:none!important;background:transparent;}
.stTabs [aria-selected="true"]{background:var(--primary)!important;color:#fff!important;}

/* Buttons */
button[kind="primary"]{background:var(--primary)!important;border:none!important;border-radius:10px!important;
padding:10px 28px!important;font-weight:700!important;color:#fff!important;
box-shadow:0 4px 14px rgba(5,150,105,0.3)!important;transition:all 0.2s!important;}
button[kind="primary"]:hover{background:var(--primary-dark)!important;transform:translateY(-1px)!important;}

/* Inputs */
.stSelectbox>div>div{background:var(--card)!important;border-color:var(--border)!important;border-radius:10px!important;}
.stSlider>div>div>div{color:var(--primary)!important;}

/* Headings */
h1,h2,h3{color:var(--text)!important;}

/* Tables */
table{width:100%;border-collapse:separate;border-spacing:0;background:var(--card);border-radius:12px;
font-size:0.88rem;margin:12px 0;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,0.04);border:1px solid var(--border);}
table th{background:var(--primary)!important;color:#fff!important;padding:12px 16px;text-align:left;font-weight:600;font-size:0.82rem;text-transform:uppercase;letter-spacing:0.5px;}
table td{padding:10px 16px;border-bottom:1px solid var(--border);color:var(--text);}
table tr:last-child td{border-bottom:none;}
table tr:hover td{background:#f8fdfb;}

/* Disclaimer */
.disclaimer{background:#fffbeb;border:1px solid #fde68a;border-radius:10px;padding:12px 16px;
font-size:0.82rem;color:#92400e;margin:12px 0;}
</style>
""", unsafe_allow_html=True)


# ── Data & Models (cached) ──
@st.cache_data(show_spinner=False)
def get_heart_data(): return load_heart_data()
@st.cache_data(show_spinner=False)
def get_diabetes_data(): return load_diabetes_data()
@st.cache_data(show_spinner=False)
def get_cancer_data(): return load_cancer_data()

@st.cache_resource(show_spinner=False)
def train_heart():
    df = get_heart_data(); return train_and_evaluate(df.drop('target',1), df['target'])
@st.cache_resource(show_spinner=False)
def train_diabetes():
    df = get_diabetes_data(); return train_and_evaluate(df.drop('Outcome',1), df['Outcome'])
@st.cache_resource(show_spinner=False)
def train_cancer():
    df, fn = get_cancer_data(); return train_and_evaluate(df.drop('target',1), df['target']), fn

# ── Chart Helpers ──
CHART_COLORS = ['#059669','#0ea5e9','#f59e0b','#ef4444']
def chart_layout(fig, h=380):
    fig.update_layout(template='plotly_white', paper_bgcolor='#fff', plot_bgcolor='#f4f9f6',
        font=dict(family='Plus Jakarta Sans',color='#1e293b'), height=h, margin=dict(l=40,r=30,t=45,b=35))
    return fig

def roc_chart(results):
    fig = go.Figure()
    for i,(name,r) in enumerate(results.items()):
        if r['roc_data']:
            fig.add_trace(go.Scatter(x=r['roc_data']['fpr'],y=r['roc_data']['tpr'],
                name=f"{name} ({r['roc_data']['auc']:.3f})",line=dict(color=CHART_COLORS[i%4],width=2.5)))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],line=dict(dash='dot',color='#cbd5e1',width=1),showlegend=False))
    fig.update_layout(title='ROC Curve Comparison',xaxis_title='False Positive Rate',yaxis_title='True Positive Rate')
    return chart_layout(fig)

def cm_chart(cm):
    fig = px.imshow(cm,x=['Negative','Positive'],y=['Negative','Positive'],text_auto=True,
        color_continuous_scale=[[0,'#d1fae5'],[0.5,'#34d399'],[1,'#047857']])
    fig.update_layout(title='Confusion Matrix')
    return chart_layout(fig, 340)

def feat_chart(imp, names, n=10):
    idx = np.argsort(imp)[-n:]
    fig = go.Figure(go.Bar(x=imp[idx],y=[names[i] for i in idx],orientation='h',
        marker=dict(color=imp[idx],colorscale=[[0,'#a7f3d0'],[1,'#047857']])))
    fig.update_layout(title=f'Top {n} Feature Importance')
    return chart_layout(fig, 400)

def show_metrics(results):
    cols = st.columns(len(results))
    for i,(name,r) in enumerate(results.items()):
        with cols[i]:
            st.markdown(f'''<div class="stat-card">
                <div class="stat-value">{r["accuracy"]*100:.1f}%</div>
                <div class="stat-label">{name}</div>
            </div>''', unsafe_allow_html=True)
    st.markdown("")
    df = pd.DataFrame({n:{'Accuracy':f"{r['accuracy']*100:.1f}%",'Precision':f"{r['precision']*100:.1f}%",
        'Recall':f"{r['recall']*100:.1f}%",'F1 Score':f"{r['f1']*100:.1f}%",
        'CV Mean':f"{r['cv_mean']*100:.1f}%"} for n,r in results.items()})
    st.markdown(df.to_html(), unsafe_allow_html=True)

def show_prediction(pred, prob):
    if pred == 1:
        st.markdown(f'''<div class="result-card result-danger">
            <h2>⚠️ Risk Detected</h2>
            <p>Model confidence: <strong>{prob[1]*100:.1f}%</strong> — Please consult a specialist immediately.</p>
        </div>''', unsafe_allow_html=True)
    else:
        st.markdown(f'''<div class="result-card result-safe">
            <h2>✅ Low Risk</h2>
            <p>Model confidence: <strong>{prob[0]*100:.1f}%</strong> — Continue maintaining a healthy lifestyle.</p>
        </div>''', unsafe_allow_html=True)


# ── Sidebar ──
with st.sidebar:
    st.markdown("# 🏥 MedPredict AI")
    st.markdown("---")
    page = st.radio("Navigate", ["🏠 Dashboard","❤️ Heart Disease","🩸 Diabetes","🔬 Breast Cancer"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div class="disclaimer">⚠️ <strong>Disclaimer:</strong> This tool is for <strong>educational purposes only</strong>. Always consult a qualified medical professional for health decisions.</div>', unsafe_allow_html=True)
    st.markdown("**Tech Stack:** scikit-learn, Plotly, Streamlit")


# ═══════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown('''<div class="hero-banner">
        <h1>🏥 MedPredict AI</h1>
        <p>Advanced Multi-Disease Prediction System powered by Machine Learning</p>
    </div>''', unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    cards = [("❤️","Heart Disease","Cardiovascular Risk",c1),("🩸","Diabetes","Glucose Analysis",c2),("🔬","Breast Cancer","Tumor Detection",c3)]
    for icon,title,sub,col in cards:
        with col:
            st.markdown(f'''<div class="stat-card">
                <div class="stat-icon">{icon}</div>
                <div class="stat-value" style="font-size:1.3rem">{title}</div>
                <div class="stat-label">{sub}</div>
            </div>''', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-hdr"><h3>🚀 How It Works</h3></div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    steps = [("1️⃣","Select Disease","Choose from sidebar",c1),("2️⃣","Input Data","Enter patient info",c2),
             ("3️⃣","Get Prediction","Instant ML results",c3),("4️⃣","View Analytics","ROC, metrics & more",c4)]
    for icon,title,desc,col in steps:
        with col:
            st.markdown(f'''<div class="stat-card"><div class="stat-icon">{icon}</div>
                <div style="font-weight:700;font-size:0.95rem;color:#1e293b">{title}</div>
                <div class="stat-label" style="text-transform:none;letter-spacing:0">{desc}</div>
            </div>''', unsafe_allow_html=True)

    st.markdown('<div class="section-hdr"><h3>🧠 Models Used</h3></div>', unsafe_allow_html=True)
    models_desc = [("Random Forest","Ensemble of 100 decision trees"),("Gradient Boosting","Sequential boosting"),
                   ("SVM (RBF)","Support vector classification"),("Logistic Regression","Linear baseline")]
    mc = st.columns(4)
    for i,(n,d) in enumerate(models_desc):
        with mc[i]:
            st.markdown(f'''<div class="stat-card"><div style="font-weight:700;color:#047857;font-size:0.95rem">{n}</div>
                <div class="stat-label" style="text-transform:none;letter-spacing:0">{d}</div>
            </div>''', unsafe_allow_html=True)


# ═══════════════════════════════════════
#  HEART DISEASE
# ═══════════════════════════════════════
elif page == "❤️ Heart Disease":
    st.markdown('''<div class="hero-banner">
        <h1>❤️ Heart Disease Prediction</h1>
        <p>Cleveland Heart Disease Dataset — 13 clinical features, 303 patients</p>
    </div>''', unsafe_allow_html=True)

    with st.spinner("Training models..."): results, scaler = train_heart()
    df = get_heart_data()
    fnames = [c for c in df.columns if c!='target']
    tab1,tab2,tab3 = st.tabs(["🔮 Predict","📊 Model Analytics","📁 Dataset"])

    with tab1:
        st.markdown('<div class="section-hdr"><h3>Patient Information</h3></div>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        with c1:
            age=st.slider("Age",20,90,55); sex=st.selectbox("Sex",[("Male",1),("Female",0)],format_func=lambda x:x[0])
            cp=st.selectbox("Chest Pain",[(0,"Typical Angina"),(1,"Atypical"),(2,"Non-Anginal"),(3,"Asymptomatic")],format_func=lambda x:x[1])
            trestbps=st.slider("Resting BP (mmHg)",90,200,130); chol=st.slider("Cholesterol (mg/dl)",100,600,240)
        with c2:
            fbs=st.selectbox("Fasting Blood Sugar>120",[("No",0),("Yes",1)],format_func=lambda x:x[0])
            restecg=st.selectbox("Resting ECG",[(0,"Normal"),(1,"ST-T Abnormality"),(2,"LV Hypertrophy")],format_func=lambda x:x[1])
            thalach=st.slider("Max Heart Rate",60,220,150)
            exang=st.selectbox("Exercise Angina",[("No",0),("Yes",1)],format_func=lambda x:x[0])
        with c3:
            oldpeak=st.slider("ST Depression",0.0,6.2,1.0,0.1)
            slope=st.selectbox("ST Slope",[(0,"Upsloping"),(1,"Flat"),(2,"Downsloping")],format_func=lambda x:x[1])
            ca=st.slider("Major Vessels (0-3)",0,3,0)
            thal=st.selectbox("Thalassemia",[(0,"Normal"),(1,"Fixed Defect"),(2,"Reversible"),(3,"Other")],format_func=lambda x:x[1])
        model_choice=st.selectbox("Model",list(results.keys()),key="h_m")
        if st.button("🔮 Predict Heart Disease",type="primary",use_container_width=True):
            feat=[age,sex[1],cp[0],trestbps,chol,fbs[1],restecg[0],thalach,exang[1],oldpeak,slope[0],ca,thal[0]]
            pred,prob=predict_single(results[model_choice]['model'],scaler,feat)
            show_prediction(pred,prob)

    with tab2:
        st.markdown('<div class="section-hdr"><h3>Model Comparison</h3></div>', unsafe_allow_html=True)
        show_metrics(results)
        c1,c2=st.columns(2)
        with c1: st.plotly_chart(roc_chart(results),use_container_width=True)
        with c2:
            sel=st.selectbox("Show confusion matrix for:",list(results.keys()),key="h_cm")
            st.plotly_chart(cm_chart(results[sel]['confusion_matrix']),use_container_width=True)
        for n,r in results.items():
            if r['feature_importance'] is not None:
                st.plotly_chart(feat_chart(r['feature_importance'],fnames),use_container_width=True); break

    with tab3:
        st.markdown(f"**Dataset:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.markdown(df.head(20).to_html(), unsafe_allow_html=True)


# ═══════════════════════════════════════
#  DIABETES
# ═══════════════════════════════════════
elif page == "🩸 Diabetes":
    st.markdown('''<div class="hero-banner">
        <h1>🩸 Diabetes Prediction</h1>
        <p>Pima Indians Diabetes Dataset — 8 diagnostic features, 768 patients</p>
    </div>''', unsafe_allow_html=True)

    with st.spinner("Training models..."): results, scaler = train_diabetes()
    df = get_diabetes_data()
    fnames = [c for c in df.columns if c!='Outcome']
    tab1,tab2,tab3 = st.tabs(["🔮 Predict","📊 Model Analytics","📁 Dataset"])

    with tab1:
        st.markdown('<div class="section-hdr"><h3>Patient Information</h3></div>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            preg=st.slider("Pregnancies",0,17,1); glucose=st.slider("Glucose (mg/dl)",40,200,120)
            bp=st.slider("Blood Pressure (mmHg)",20,130,72); skin=st.slider("Skin Thickness (mm)",0,100,25)
        with c2:
            insulin=st.slider("Insulin (mu U/ml)",0,850,80); bmi=st.slider("BMI",15.0,70.0,28.0,0.1)
            dpf=st.slider("Diabetes Pedigree Function",0.05,2.5,0.5,0.01); age_d=st.slider("Age",18,90,30,key="d_age")
        model_choice=st.selectbox("Model",list(results.keys()),key="d_m")
        if st.button("🔮 Predict Diabetes",type="primary",use_container_width=True):
            pred,prob=predict_single(results[model_choice]['model'],scaler,[preg,glucose,bp,skin,insulin,bmi,dpf,age_d])
            show_prediction(pred,prob)

    with tab2:
        st.markdown('<div class="section-hdr"><h3>Model Comparison</h3></div>', unsafe_allow_html=True)
        show_metrics(results)
        c1,c2=st.columns(2)
        with c1: st.plotly_chart(roc_chart(results),use_container_width=True)
        with c2:
            sel=st.selectbox("Show confusion matrix for:",list(results.keys()),key="d_cm")
            st.plotly_chart(cm_chart(results[sel]['confusion_matrix']),use_container_width=True)
        for n,r in results.items():
            if r['feature_importance'] is not None:
                st.plotly_chart(feat_chart(r['feature_importance'],fnames),use_container_width=True); break

    with tab3:
        st.markdown(f"**Dataset:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.markdown(df.head(20).to_html(), unsafe_allow_html=True)


# ═══════════════════════════════════════
#  BREAST CANCER
# ═══════════════════════════════════════
elif page == "🔬 Breast Cancer":
    st.markdown('''<div class="hero-banner">
        <h1>🔬 Breast Cancer Detection</h1>
        <p>Wisconsin Breast Cancer Dataset — 30 cell nucleus features, 569 samples</p>
    </div>''', unsafe_allow_html=True)

    with st.spinner("Training models..."): (results,scaler),feat_names = train_cancer()
    df,_ = get_cancer_data()
    tab1,tab2,tab3 = st.tabs(["🔮 Predict","📊 Model Analytics","📁 Dataset"])

    with tab1:
        st.markdown('<div class="section-hdr"><h3>Tumor Measurements (FNA Biopsy)</h3></div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Enter cell nucleus measurements from Fine Needle Aspirate biopsy. Non-specified features use median values.</div>', unsafe_allow_html=True)
        key_f=['mean radius','mean texture','mean perimeter','mean area','mean smoothness',
               'mean compactness','mean concavity','mean concave points','mean symmetry','mean fractal dimension']
        defs=df[key_f].median().to_dict()
        vals={}
        c1,c2=st.columns(2)
        for i,f in enumerate(key_f):
            col=c1 if i<5 else c2
            with col:
                mn,mx=float(df[f].min()),float(df[f].max())
                vals[f]=st.slider(f.replace('mean ','').title(),mn,mx,float(defs[f]),(mx-mn)/100)
        all_v=[vals.get(f,float(df[f].median())) for f in feat_names]
        model_choice=st.selectbox("Model",list(results.keys()),key="c_m")
        if st.button("🔮 Detect Cancer",type="primary",use_container_width=True):
            pred,prob=predict_single(results[model_choice]['model'],scaler,all_v)
            show_prediction(pred,prob)

    with tab2:
        st.markdown('<div class="section-hdr"><h3>Model Comparison</h3></div>', unsafe_allow_html=True)
        show_metrics(results)
        c1,c2=st.columns(2)
        with c1: st.plotly_chart(roc_chart(results),use_container_width=True)
        with c2:
            sel=st.selectbox("Show confusion matrix for:",list(results.keys()),key="c_cm")
            st.plotly_chart(cm_chart(results[sel]['confusion_matrix']),use_container_width=True)
        for n,r in results.items():
            if r['feature_importance'] is not None:
                st.plotly_chart(feat_chart(r['feature_importance'],feat_names,15),use_container_width=True); break

    with tab3:
        st.markdown(f"**Dataset:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.markdown(df.head(20).to_html(), unsafe_allow_html=True)
