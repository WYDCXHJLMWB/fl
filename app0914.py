import streamlit as st
import pandas as pd
import bcrypt
import os
import joblib
import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer
import warnings
from datetime import datetime
import re

warnings.filterwarnings('ignore')

# --------------------- 用户认证模块 ---------------------
USERS_FILE = "users.csv"
if not os.path.exists(USERS_FILE):
    pd.DataFrame(columns=["username", "password_hash", "email"]).to_csv(USERS_FILE, index=False)

def load_users():
    return pd.read_csv(USERS_FILE)

def save_user(username, password, email):
    users = load_users()
    if username in users['username'].values:
        return False
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    new_user = pd.DataFrame([[username, password_hash.decode(), email]],
                          columns=["username", "password_hash", "email"])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv(USERS_FILE, index=False)
    return True

def verify_user(username, password):
    users = load_users()
    user = users[users['username'] == username]
    if user.empty:
        return False
    return bcrypt.checkpw(password.encode(), user.iloc[0]['password_hash'].encode())

def reset_password_by_email(email, new_password):
    users = load_users()
    user = users[users['email'] == email]
    if not user.empty:
        password_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
        users.loc[users['email'] == email, 'password_hash'] = password_hash
        users.to_csv(USERS_FILE, index=False)
        return True
    return False

# --------------------- 全局状态 ---------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = None
if 'input_values' not in st.session_state:
    st.session_state.input_values = {}
if 'inverse_results' not in st.session_state:
    st.session_state.inverse_results = None

# --------------------- 样式配置 ---------------------
def apply_global_styles():
    st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <style>
        .stApp { background-color: #f8f9fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .global-header h1 { color: #1e3d59; margin-bottom: 0.5rem; font-size: 2.8rem !important; text-align: center; }
        .global-header p { color: #4a6572; font-size: 1.5rem !important; margin-top: 0; text-align: center; }
        .feature-card { background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 1.5rem; border-left: 4px solid #3f87a6; }
        .stTextInput input, .stNumberInput input, .stSelectbox select { padding: 12px 16px !important; font-size: 16px !important; border-radius: 8px !important; }
        .stButton button { background-color: #3f87a6 !important; color: white !important; border-radius: 8px !important; padding: 10px 20px !important; font-weight: 500 !important; transition: all 0.3s ease !important; }
        .stButton button:hover { background-color: #2c6a8a !important; }
        footer { margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #eaeaea; color: #6c757d; font-size: 0.9rem; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

def render_global_header():
    st.markdown("""
    <div class="global-header">
        <h1>阻燃聚合物复合材料智能设计平台</h1>
        <p>Flame Retardant Composites AI Platform</p>
    </div>
    """, unsafe_allow_html=True)

# --------------------- 首页内容 ---------------------
def show_homepage():
    apply_global_styles()
    render_global_header()
    
    st.markdown("""<div style="max-width:1400px; margin:0 auto; padding:2rem;">""", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:1.2rem; line-height:1.6; margin-bottom:2.5rem; text-align: center;">
        🚀 本平台融合AI与材料科学技术，致力于高分子复合材料的智能化设计，
        重点关注阻燃性能、力学性能和热稳定性的多目标优化与调控。
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h2 style="font-size:1.8rem; color:#1e3d59; border-bottom: 2px solid #3f87a6; padding-bottom:0.5rem; margin-bottom:1.5rem;">
        🌟 核心功能
    </h2>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3 style="font-size:1.4rem; color:#1e3d59; margin:0 0 1rem 0;">🔥 智能性能预测</h3>
            <p style="font-size:1.1rem;">• 支持LOI（极限氧指数）预测<br>• TS（拉伸强度）预测<br></p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3 style="font-size:1.4rem; color:#1e3d59; margin:0 0 1rem 0;">⚗️ 配方优化系统</h3>
            <p style="font-size:1.1rem;">• 根据输入目标推荐配方<br>• 支持选择配方种类<br>• 添加剂比例智能推荐<br>• 多目标逆向设计(PHRR/LOI)</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <h2 style="font-size:1.8rem; color:#1e3d59; border-bottom: 2px solid #3f87a6; padding-bottom:0.5rem; margin-bottom:1.5rem;">
        🏆 研究成果
    </h2>
    <div class="feature-card">
        <p style="font-size:1.1rem;">
            Ma Weibin, Li Ling, Zhang Yu, et al.<br>
            <em>Active learning-based generative design of halogen-free flame-retardant polymeric composites.</em><br>
            <strong>Journal of Materials Informatics</strong> 2025;5:09.<br>
            DOI: <a href="https://doi.org/10.20517/jmi.2025.09" target="_blank" style="color:#3f87a6; text-decoration:underline;">10.20517/jmi.2025.09</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h2 style="font-size:1.8rem; color:#1e3d59; border-bottom: 2px solid #3f87a6; padding-bottom:0.5rem; margin-bottom:1.5rem;">
        👨💻 开发团队
    </h2>
    """, unsafe_allow_html=True)
    
    col_dev, col_sup = st.columns(2)
    with col_dev:
        st.markdown("""
        <div class="feature-card">
            <p style="font-size:1.1rem;">
                上海大学功能高分子<br>PolyDesign<br>马维宾 | 李凌 | 张瑜<br>宋娜 | 丁鹏<br>
                上海大学计算机学院<br>韩越兴 | 李睿杰<br>
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_sup:
        st.markdown("""
        <div class="feature-card">
            <h3 style="font-size:1.4rem; color:#1e3d59; margin:0 0 1rem 0;">🙏 项目支持</h3>
            <p style="font-size:1.1rem;">云南省科技重点计划<br>项目编号：202302AB080022<br></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""<div style="margin-top: 3rem; background: #ffffff; padding: 2rem; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">""", unsafe_allow_html=True)
    st.markdown('<h2 style="font-size:1.8rem; color:#1e3d59; text-align:center; margin-bottom:1.5rem;">🔐 用户认证</h2>', unsafe_allow_html=True)
    
    tab_login, tab_register, tab_forgot = st.tabs(["登录", "注册", "忘记密码"])

    with tab_login:
        with st.form("login_form", clear_on_submit=True):
            username = st.text_input("用户名", key="login_user")
            password = st.text_input("密码", type="password", key="login_pwd")
            if st.form_submit_button("登录", use_container_width=True):
                if not all([username, password]):
                    st.error("请输入用户名和密码")
                elif verify_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.user = username
                    st.rerun()
                else:
                    st.error("用户名或密码错误")

    with tab_register:
        with st.form("register_form", clear_on_submit=True):
            new_user = st.text_input("用户名（4-20位字母数字）", key="reg_user").strip()
            new_pwd = st.text_input("设置密码（至少6位）", type="password", key="reg_pwd")
            confirm_pwd = st.text_input("确认密码", type="password", key="reg_pwd_confirm")
            email = st.text_input("电子邮箱", key="reg_email")
            if st.form_submit_button("注册", use_container_width=True):
                if new_pwd != confirm_pwd:
                    st.error("两次密码输入不一致")
                elif len(new_user) < 4 or not new_user.isalnum():
                    st.error("用户名格式不正确")
                elif len(new_pwd) < 6:
                    st.error("密码长度至少6个字符")
                elif "@" not in email:
                    st.error("请输入有效邮箱地址")
                else:
                    if save_user(new_user, new_pwd, email):
                        st.success("注册成功！请登录")
                    else:
                        st.error("用户名已存在")

    with tab_forgot:
        with st.form("forgot_form", clear_on_submit=True):
            email = st.text_input("注册邮箱", key="reset_email")
            new_password = st.text_input("新密码", type="password", key="new_pwd")
            confirm_password = st.text_input("确认密码", type="password", key="confirm_pwd")
            if st.form_submit_button("重置密码", use_container_width=True):
                if not all([email, new_password, confirm_password]):
                    st.error("请填写所有字段")
                elif new_password != confirm_password:
                    st.error("两次输入密码不一致")
                elif reset_password_by_email(email, new_password):
                    st.success("密码已重置，请使用新密码登录")
                else:
                    st.error("该邮箱未注册")
    
    st.markdown("</div></div>", unsafe_allow_html=True)

# --------------------- 预测相关类和函数 ---------------------
class Predictor:
    def __init__(self, scaler_path, svc_path):
        self.scaler = joblib.load(scaler_path)
        self.model = joblib.load(svc_path)
        self.static_cols = ["产品质量指标_Sn%", "添加比例", "一甲%"]
        self.time_series_cols = ["黄度值_3min", "6min", "9min", "12min", "15min", "18min", "21min", "24min"]
        self.eng_features = ['seq_length', 'max_value', 'mean_value', 'min_value',
                             'std_value', 'trend', 'range_value', 'autocorr']
        self.imputer = SimpleImputer(strategy="mean")

    def _truncate(self, df):
        time_cols = [col for col in df.columns if "min" in col.lower()]
        time_cols_ordered = [col for col in df.columns if col in time_cols]
        if time_cols_ordered:
            row = df.iloc[0][time_cols_ordered]
            if row.notna().any():
                max_idx = row.idxmax()
                max_pos = time_cols_ordered.index(max_idx)
                for col in time_cols_ordered[max_pos + 1:]:
                    df.at[df.index[0], col] = np.nan
        return df

    def _get_slope(self, row, col=None):
        x = np.arange(len(row))
        y = row.values
        mask = ~np.isnan(y)
        if sum(mask) >= 2:
            return stats.linregress(x[mask], y[mask])[0]
        return np.nan

    def _calc_autocorr(self, row):
        values = row.dropna().values
        if len(values) > 1:
            n = len(values)
            mean = np.mean(values)
            numerator = sum((values[:-1] - mean) * (values[1:] - mean))
            denominator = sum((values - mean) ** 2)
            if denominator != 0:
                return numerator / denominator
        return np.nan

    def _extract_time_series_features(self, df):
        time_data = df[self.time_series_cols]
        time_data_filled = time_data.ffill(axis=1)
        features = pd.DataFrame()
        features['seq_length'] = time_data_filled.notna().sum(axis=1)
        features['max_value'] = time_data_filled.max(axis=1)
        features['mean_value'] = time_data_filled.mean(axis=1)
        features['min_value'] = time_data_filled.min(axis=1)
        features['std_value'] = time_data_filled.std(axis=1)
        features['range_value'] = features['max_value'] - features['min_value']
        features['trend'] = time_data_filled.apply(self._get_slope, axis=1)
        features['autocorr'] = time_data_filled.apply(self._calc_autocorr, axis=1)
        return features

    def predict_one(self, sample):
        full_cols = self.static_cols + self.time_series_cols
        df = pd.DataFrame([sample], columns=full_cols)
        df = self._truncate(df)
        static_features = df[self.static_cols]
        time_features = self._extract_time_series_features(df)
        feature_df = pd.concat([static_features, time_features], axis=1)
        feature_df = feature_df[self.static_cols + self.eng_features]
        if feature_df.shape[1] != self.scaler.n_features_in_:
            raise ValueError(f"特征维度不匹配！当前：{feature_df.shape[1]}，需要：{self.scaler.n_features_in_}")
        X_scaled = self.scaler.transform(feature_df)
        return self.model.predict(X_scaled)[0]


# =====================================================================
#                    PHRR 模型 / 特征 加载与预测
# =====================================================================
PHRR_MODEL_PATH = "catboost_regressor_143232.pkl"
PHRR_FEATURES_TXT = "selected_features.txt"

PHRR_VARIABLE_RANGES = {
    'PAPP': (19, 24),
    'MPP': (8.5, 14),
    'ZS': (0.4, 1.6),
    'W': (2.5, 9)
}
ADP_FIXED = 0.3


def load_phrr_features():
    features = []
    if not os.path.exists(PHRR_FEATURES_TXT):
        return features
    try:
        with open(PHRR_FEATURES_TXT, 'r', encoding='utf-8') as f:
            content = f.read()

        if '选中数值特征列表:' in content:
            start_idx = content.find('选中数值特征列表:')
            feature_section = content[start_idx + len('选中数值特征列表:'):].strip()
            for line in feature_section.split('\n'):
                line = line.strip()
                if not line:
                    continue
                match = re.match(r'^\d+\.\s*(.+)$', line)
                if match:
                    features.append(match.group(1))
                elif line and not line.startswith('=') and not line.startswith('--'):
                    if not any(k in line for k in ['实验随机种子', '核心特征', '总数值特征数', '选中特征数量']):
                        features.append(line)

        if not features:
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                match = re.match(r'^\d+\.\s*(.+)$', line)
                if match:
                    features.append(match.group(1))

        clean_features = []
        for f in features:
            f = f.strip()
            if re.match(r'^[A-Za-z0-9_\-/\.]+$', f):
                clean_features.append(f)
        features = clean_features

    except Exception as e:
        st.warning(f"PHRR 特征文件解析失败: {e}")
    return features


@st.cache_resource
def load_phrr_model():
    try:
        if os.path.exists(PHRR_MODEL_PATH):
            model = joblib.load(PHRR_MODEL_PATH)
            return model
        else:
            st.error(f"❌ 模型文件不存在：{PHRR_MODEL_PATH}")
    except Exception as e:
        st.error(f"PHRR 模型加载失败: {e}")
    return None


def feature_engineering_phrr(p, phrr_features):
    df = pd.DataFrame({k: [v] for k, v in p.items()})

    for base in ['PP', 'PAPP', 'MPP', 'W', 'ZS', 'ADP']:
        if base not in df.columns:
            df[base] = 0.0

    additive_cols = ['PAPP', 'MPP', 'W', 'ZS', 'ADP']
    existing_additive = [col for col in additive_cols if col in df.columns]
    if existing_additive:
        df['total_ad'] = df[existing_additive].sum(axis=1)
        max_val = df[existing_additive].max(axis=1)
        df['additive_ratio'] = df['total_ad'] / (max_val + 1e-6)
    else:
        df['total_ad'] = 0
        df['additive_ratio'] = 0

    df['PAPP_ratio'] = df['PAPP'] / (df['total_ad'] + 1e-6) if 'PAPP' in df.columns else 0
    df['MPP_ratio'] = df['MPP'] / (df['total_ad'] + 1e-6) if 'MPP' in df.columns else 0

    synergist_cols = ['W', 'ZS', 'ADP']
    existing_synergist = [col for col in synergist_cols if col in df.columns]
    if existing_synergist:
        df['synergist_total'] = df[existing_synergist].sum(axis=1)
        df['W_ratio_in_synergist'] = df['W'] / (df['synergist_total'] + 1e-6) if 'W' in df.columns else 0
    else:
        df['synergist_total'] = 0
        df['W_ratio_in_synergist'] = 0

    df['PAPP_MPP_W_interaction'] = df['PAPP'] * df['MPP'] * df['W'] if all(c in df.columns for c in ['PAPP','MPP','W']) else 0
    df['PAPP_square'] = df['PAPP'] ** 2 if 'PAPP' in df.columns else 0
    df['MPP_square'] = df['MPP'] ** 2 if 'MPP' in df.columns else 0
    df['PAPP_MPP_ratio'] = df['PAPP'] / (df['MPP'] + 1e-6) if all(c in df.columns for c in ['PAPP','MPP']) else 0
    df['PAPP_W_ratio'] = df['PAPP'] / (df['W'] + 1e-6) if all(c in df.columns for c in ['PAPP','W']) else 0
    df['PAPP_sqrt'] = np.sqrt(np.clip(df['PAPP'], 0, None)) if 'PAPP' in df.columns else 0
    df['MPP_sqrt'] = np.sqrt(np.clip(df['MPP'], 0, None)) if 'MPP' in df.columns else 0
    df['ZS_ratio'] = df['ZS'] / (df['total_ad'] + 1e-6) if all(c in df.columns for c in ['ZS','total_ad']) else 0
    df['W_ratio'] = df['W'] / (df['total_ad'] + 1e-6) if all(c in df.columns for c in ['W','total_ad']) else 0
    df['ADP_ratio'] = df['ADP'] / (df['total_ad'] + 1e-6) if all(c in df.columns for c in ['ADP','total_ad']) else 0

    for feat in phrr_features:
        if feat not in df.columns:
            df[feat] = 0.0

    df = df[phrr_features]
    return df


def predict_phrr(p, phrr_model, phrr_features):
    try:
        df_eng = feature_engineering_phrr(p, phrr_features)
        x = df_eng.values.astype(np.float64)
        if x.shape[1] == 0:
            return 9999.0

        try:
            pred = phrr_model.predict(x)[0]
            return float(pred)
        except Exception:
            if hasattr(phrr_model, 'n_features_in_') and phrr_model.n_features_in_ > 0:
                n_expected = phrr_model.n_features_in_
                if x.shape[1] != n_expected:
                    if x.shape[1] > n_expected:
                        x = x[:, :n_expected]
                    else:
                        padding = np.zeros((x.shape[0], n_expected - x.shape[1]))
                        x = np.hstack([x, padding])
                    pred = phrr_model.predict(x)[0]
                    return float(pred)
            raise
    except Exception as e:
        print(f"[predict_phrr] error: {e}")
        return 9999.0


# --------------------- LOI 预测 ---------------------
_LOI_MATRIX = ["PP", "PA", "PC/ABS", "POM", "PBT", "PVC"]
_LOI_FR = ["AHP", "CFA", "ammonium octamolybdate", "Al(OH)3", "APP",
           "Pentaerythritol", "DOPO", "XS-FR-8310", "ZS", "XiuCheng", "ZHS",
           "ZnB", "antimony oxides", "Mg(OH)2", "TCA", "MPP", "PAPP", "其他"]
_LOI_ADD = ["Anti-drip-agent", "ZBS-PV-OA", "FP-250S",
            "wollastonite", "SiO2", "silane coupling agent",
            "antioxidant", "M-2200B", "Custom Additive"]

LOI_ALL_FEATURES = sorted(_LOI_MATRIX + _LOI_FR + _LOI_ADD)


def _build_loi_vector(p, n_expected):
    vec = [float(p.get(f, 0.0)) for f in LOI_ALL_FEATURES]
    if len(vec) < n_expected:
        vec += [0.0] * (n_expected - len(vec))
    else:
        vec = vec[:n_expected]
    return vec


def predict_loi(p, loi_model, loi_scaler):
    try:
        n_expected = getattr(loi_scaler, 'n_features_in_', 25)
        vec = _build_loi_vector(p, n_expected)
        x = np.array([vec], dtype=np.float64)
        x_scaled = loi_scaler.transform(x)
        pred = loi_model.predict(x_scaled)[0]
        pred = max(17.0, min(pred, 50.0))
        return float(pred)
    except Exception as e:
        print(f"[predict_loi] error: {e}")
        return 9999.0


# --------------------- 逆向设计主界面 ---------------------
def render_inverse_design_page(models):
    st.subheader("🎯 配方逆向优化（PHRR & LOI）")
    st.markdown("""
    输入你期望达到的 **PHRR（热释放速率峰值）** 和 **LOI（极限氧指数）** 目标值，
    系统会自动搜索出最接近目标的 **3 个最优配方**。
    所有配方各组分之和严格等于 100。
    """)

    phrr_model = load_phrr_model()
    phrr_features = load_phrr_features()

    if phrr_model is None:
        st.error(f"❌ PHRR 模型加载失败。请确认 `{PHRR_MODEL_PATH}` 已在仓库根目录。")
        return

    if len(phrr_features) == 0:
        st.error(f"❌ 未从 `{PHRR_FEATURES_TXT}` 解析到特征列表，请检查文件内容格式。")
        return

    if models is None or models.get("loi_model") is None:
        st.error("❌ LOI 模型不可用，无法进行逆向设计。")
        return

    loi_model = models["loi_model"]
    loi_scaler = models["loi_scaler"]

    # ---- 参数输入 ----
    st.markdown("### ⚙️ 优化参数设置")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        target_phrr = st.number_input(
            "目标 PHRR (kW/m²)", min_value=50.0, max_value=1000.0,
            value=200.0, step=10.0, format="%.1f"
        )
    with col_t2:
        target_loi = st.number_input(
            "目标 LOI (%)", min_value=17.0, max_value=50.0,
            value=28.0, step=0.5, format="%.1f"
        )

    max_evals = st.number_input(
        "搜索精细度（推荐 1000–5000）",
        min_value=50, max_value=20000, value=2000, step=100
    )

    st.markdown("### 📐 变量范围")
    st.caption("说明：本次搜索变量包括 PAPP、MPP、ZS、W，PP 由总和 100 自动补齐，ADP 固定为 0.3。")

    use_default_ranges = st.checkbox("使用推荐的取值范围", value=True)
    if use_default_ranges:
        ranges = PHRR_VARIABLE_RANGES
    else:
        ranges = {}
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            papp_min = st.number_input("PAPP 最小", value=19.0, step=0.5)
            papp_max = st.number_input("PAPP 最大", value=24.0, step=0.5)
            ranges['PAPP'] = (papp_min, papp_max)
        with c2:
            mpp_min = st.number_input("MPP 最小", value=8.5, step=0.5)
            mpp_max = st.number_input("MPP 最大", value=14.0, step=0.5)
            ranges['MPP'] = (mpp_min, mpp_max)
        with c3:
            zs_min = st.number_input("ZS 最小", value=0.4, step=0.1)
            zs_max = st.number_input("ZS 最大", value=1.6, step=0.1)
            ranges['ZS'] = (zs_min, zs_max)
        with c4:
            w_min = st.number_input("W 最小", value=2.5, step=0.5)
            w_max = st.number_input("W 最大", value=9.0, step=0.5)
            ranges['W'] = (w_min, w_max)

    if st.button("🚀 开始优化配方", type="primary", use_container_width=True):
        # 自检
        test_p = {'PP': 75.0, 'PAPP': 21.0, 'MPP': 11.0, 'ZS': 1.0, 'W': 5.0, 'ADP': ADP_FIXED}
        test_phrr = predict_phrr(test_p, phrr_model, phrr_features)
        test_loi = predict_loi(test_p, loi_model, loi_scaler)

        if test_phrr == 9999.0 or test_loi == 9999.0:
            st.error("❌ 预测函数自检失败。请检查模型和特征文件是否正确。")
            return

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        try:
            from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
            HAS_HYPEROPT = True
        except ImportError:
            HAS_HYPEROPT = False

        history = []
        evaluated = set()
        counter = {'n': 0}

        def objective(params):
            tup = tuple(round(params[k], 4) for k in ['PAPP', 'MPP', 'ZS', 'W'])
            if tup in evaluated:
                return 9999.0
            evaluated.add(tup)

            p = params.copy()
            p['ADP'] = ADP_FIXED
            pp = 100.0 - p['PAPP'] - p['MPP'] - p['ZS'] - p['W'] - ADP_FIXED
            if pp < 50.0 or pp > 100.0:
                return 9999.0
            p['PP'] = pp

            phrr_pred = predict_phrr(p, phrr_model, phrr_features)
            loi_pred = predict_loi(p, loi_model, loi_scaler)
            if phrr_pred == 9999.0 or loi_pred == 9999.0:
                return 9999.0

            phrr_err = abs(phrr_pred - target_phrr) / max(target_phrr, 1e-6)
            loi_err = abs(loi_pred - target_loi) / max(target_loi, 1e-6)
            loss = phrr_err + loi_err

            history.append({
                'PP': p['PP'], 'PAPP': p['PAPP'], 'MPP': p['MPP'],
                'ZS': p['ZS'], 'W': p['W'], 'ADP': p['ADP'],
                'phrr_pred': phrr_pred, 'loi_pred': loi_pred,
                'phrr_error': phrr_err, 'loi_error': loi_err,
                'loss': loss
            })
            return loss

        def update_progress():
            counter['n'] += 1
            n_done = counter['n']
            valid = [h['loss'] for h in history if h['loss'] < 9999]
            best_loss = min(valid) if valid else 9999.0
            progress_bar.progress(min(n_done / max_evals, 1.0))
            status_text.text(f"已完成 {n_done}/{max_evals} 次搜索，当前最优 loss={best_loss:.4f}")

        if HAS_HYPEROPT:
            space = {
                'PAPP': hp.uniform('PAPP', ranges['PAPP'][0], ranges['PAPP'][1]),
                'MPP': hp.uniform('MPP', ranges['MPP'][0], ranges['MPP'][1]),
                'ZS': hp.uniform('ZS', ranges['ZS'][0], ranges['ZS'][1]),
                'W': hp.uniform('W', ranges['W'][0], ranges['W'][1])
            }

            def objective_hp(params):
                loss = objective(params)
                update_progress()
                return {'loss': loss, 'status': STATUS_OK}

            fmin(
                fn=objective_hp,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=Trials(),
                show_progressbar=False
            )
        else:
            rng = np.random.default_rng(42)
            for i in range(max_evals):
                params = {
                    'PAPP': rng.uniform(*ranges['PAPP']),
                    'MPP': rng.uniform(*ranges['MPP']),
                    'ZS': rng.uniform(*ranges['ZS']),
                    'W': rng.uniform(*ranges['W']),
                }
                objective(params)
                update_progress()

        progress_bar.progress(1.0)
        status_text.empty()

        if len(history) == 0:
            st.error("❌ 未找到任何有效配方，请检查变量范围或模型。")
            return

        df_res = pd.DataFrame(history)
        df_res = df_res[df_res['loss'] < 9999].sort_values('loss').reset_index(drop=True)

        # 去重
        df_res['_key'] = df_res.apply(
            lambda r: (round(r['PP'], 2), round(r['PAPP'], 2),
                       round(r['MPP'], 2), round(r['ZS'], 2), round(r['W'], 2)),
            axis=1
        )
        df_res = df_res.drop_duplicates('_key').drop(columns=['_key']).reset_index(drop=True)

        st.session_state.inverse_results = df_res

        # 只取前 3 个最优配方
        TOP_N = 3
        best = df_res.head(TOP_N).reset_index(drop=True)

        # 构造只含配方组分与预测性能的表格（不含总和、Loss）
        table_data = []
        for i in range(len(best)):
            row = best.iloc[i]
            table_data.append({
                "配方": f"配方{i+1}",
                "PP": round(float(row['PP']), 2),
                "PAPP": round(float(row['PAPP']), 2),
                "MPP": round(float(row['MPP']), 2),
                "ZS": round(float(row['ZS']), 2),
                "W": round(float(row['W']), 2),
                "ADP": round(float(row['ADP']), 2),
                "预测PHRR": round(float(row['phrr_pred']), 2),
                "预测LOI": round(float(row['loi_pred']), 2),
            })

        df_table = pd.DataFrame(table_data)

        st.markdown("### 🏆 最优配方")
        st.dataframe(df_table, hide_index=True, use_container_width=True)

        csv = df_table.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            "📥 下载配方表 (CSV)",
            data=csv,
            file_name=f"top3_formulations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )


# --------------------- 主应用逻辑 ---------------------
if st.session_state.logged_in:
    page = st.sidebar.selectbox(
        "🔧 主功能选择",
        ["性能预测", "配方建议"],
        key="main_nav"
    )
    sub_page = None
    if page == "配方建议":
        sub_page = st.sidebar.selectbox(
            "🔧 子功能选择",
            ["配方优化", "添加剂推荐"],
            key="sub_nav"
        )
    with st.sidebar:
        if st.button("🚪 退出登录", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.input_values = {}
            st.session_state.inverse_results = None
            st.rerun()

    @st.cache_resource
    def load_models():
        try:
            loi_data = joblib.load("model_and_scaler_loi.pkl")
            ts_data = joblib.load("model_and_scaler_ts1.pkl")
            return {
                "loi_model": loi_data["model"],
                "loi_scaler": loi_data["scaler"],
                "ts_model": ts_data["model"],
                "ts_scaler": ts_data["scaler"],
            }
        except Exception as e:
            st.error(f"模型加载失败: {str(e)}")
            return None

    models = load_models()

    def get_unit(fraction_type):
        if fraction_type == "质量":
            return "g"
        elif fraction_type == "质量分数":
            return "wt%"
        elif fraction_type == "体积分数":
            return "vol%"

    apply_global_styles()
    render_global_header()

    if page == "性能预测":
        st.subheader("🔮 性能预测：基于配方预测LOI和TS")

        matrix_materials = {
            "PP": {"name": "Polypropylene", "full_name": "Polypropylene (PP)", "range": (53.5, 99.5)},
            "PA": {"name": "Polyamide", "full_name": "Polyamide (PA)", "range": (0, 100)},
            "PC/ABS": {"name": "Polycarbonate/Acrylonitrile Butadiene Styrene Blend", "full_name": "Polycarbonate/Acrylonitrile Butadiene Styrene Blend (PC/ABS)", "range": (0, 100)},
            "POM": {"name": "Polyoxymethylene", "full_name": "Polyoxymethylene (POM)", "range": (0, 100)},
            "PBT": {"name": "Polybutylene Terephthalate", "full_name": "Polybutylene Terephthalate (PBT)", "range": (0, 100)},
            "PVC": {"name": "Polyvinyl Chloride", "full_name": "Polyvinyl Chloride (PVC)", "range": (0, 100)},
        }

        flame_retardants = {
            "AHP": {"name": "Aluminum Hyphosphite", "range": (0, 25)},
            "CFA": {"name": "Carbon Forming agent", "range": (0, 10)},
            "ammonium octamolybdate": {"name": "Ammonium Octamolybdate", "range": (0, 3.4)},
            "Al(OH)3": {"name": "Aluminum Hydroxide", "range": (0, 10)},
            "APP": {"name": "Ammonium Polyphosphate", "range": (0, 19.5)},
            "Pentaerythritol": {"name": "Pentaerythritol", "range": (0, 1.3)},
            "DOPO": {"name": "9,10-Dihydro-9-oxa-10-phosphaphenanthrene-10-oxide", "range": (0, 27)},
            "XS-FR-8310": {"name": "XS-FR-8310", "range": (0, 35)},
            "ZS": {"name": "Zinc Stannate", "range": (0, 34.5)},
            "XiuCheng": {"name": "XiuCheng Flame Retardant", "range": (0, 35)},
            "ZHS": {"name": "Hydroxy Zinc Stannate", "range": (0, 34.5)},
            "ZnB": {"name": "Zinc Borate", "range": (0, 2)},
            "antimony oxides": {"name": "Antimony Oxides", "range": (0, 2)},
            "Mg(OH)2": {"name": "Magnesium Hydroxide", "range": (0, 34.5)},
            "TCA": {"name": "Triazine Carbonization Agent", "range": (0, 17.4)},
            "MPP": {"name": "Melamine Polyphosphate", "range": (0, 25)},
            "PAPP": {"name": "Piperazine Pyrophosphate", "range": (0, 24.5)},
            "其他": {"name": "Other", "range": (0, 100)},
        }

        additives = {
            "processing additives": {
                "Anti-drip-agent": {"name": "Polytetrafluoroethylene Anti-dripping Agent", "range": (0, 0.3)},
                "ZBS-PV-OA": {"name": "Zinc Borate Stabilizer PV-OA Series", "range": (0, 35)},
                "FP-250S": {"name": "Processing Aid FP-250S (Acrylic)", "range": (0, 35)},
            },
            "Fillers": {
                "wollastonite": {"name": "Wollastonite (Calcium Metasilicate)", "range": (0, 5)},
                "SiO2": {"name": "Silicon Dioxide", "range": (0, 6)},
            },
            "Coupling Agents": {
                "silane coupling agent": {"name": "Amino Silane Coupling Agent", "range": (0.5, 3)},
            },
            "Antioxidants": {
                "antioxidant": {"name": "Irganox 1010 Antioxidant", "range": (0.1, 0.5)},
            },
            "Lubricants": {
                "M-2200B": {"name": "Lubricant M-2200B (Ester-based)", "range": (0.5, 3)},
            },
            "Functional Additives": {
                "Custom Additive": {"name": "Custom Additive", "range": (0, 5)},
            },
        }

        fraction_type = st.sidebar.selectbox("选择输入的单位", ["质量", "质量分数", "体积分数"])

        st.subheader("请选择配方成分")
        col_matrix = st.columns([4, 3], gap="medium")
        with col_matrix[0]:
            selected_matrix = st.selectbox("选择基体材料", [matrix_materials[key]["full_name"] for key in matrix_materials], index=0)
            matrix_key = [key for key in matrix_materials if matrix_materials[key]["full_name"] == selected_matrix][0]
            matrix_name = matrix_materials[matrix_key]["name"]
            matrix_range = matrix_materials[matrix_key]["range"]
            st.markdown(f"**推荐范围**: {matrix_range[0]} - {matrix_range[1]}")
        with col_matrix[1]:
            unit_matrix = get_unit(fraction_type)
            st.session_state.input_values[matrix_key] = st.number_input(
                f"{matrix_name} 含量 ({unit_matrix})", min_value=0.0, max_value=100.0, value=50.0, step=0.1
            )

        st.subheader("请选择阻燃剂")
        selected_flame_retardants = st.multiselect(
            "选择阻燃剂（必选锡酸锌和羟基锡酸锌）",
            [flame_retardants[key]["name"] for key in flame_retardants],
            default=[flame_retardants[list(flame_retardants.keys())[0]]["name"]]
        )
        for flame_name in selected_flame_retardants:
            for key, value in flame_retardants.items():
                if value["name"] == flame_name:
                    flame_info = value
                    with st.expander(f"{flame_info['name']} 推荐范围"):
                        st.write(f"推荐范围：{flame_info['range'][0]} - {flame_info['range'][1]}")
                        unit_add = get_unit(fraction_type)
                        min_val = float(flame_info['range'][0])
                        max_val = float(flame_info['range'][1])
                        default_value = max(min_val, 0.0)
                        st.session_state.input_values[key] = st.number_input(
                            f"{flame_info['name']} 含量 ({unit_add})",
                            min_value=min_val, max_value=max_val, value=default_value, step=0.1, key=f"fr_{key}"
                        )

        st.subheader("选择助剂")
        selected_additives = st.multiselect(
            "选择助剂（可多选）", list(additives.keys()), default=[list(additives.keys())[0]]
        )
        for category in selected_additives:
            for ad, additive_info in additives[category].items():
                with st.expander(f"{additive_info['name']} 推荐范围"):
                    st.write(f"推荐范围：{additive_info['range'][0]} - {additive_info['range'][1]}")
                    unit_additive = get_unit(fraction_type)
                    min_additive = float(additive_info["range"][0])
                    max_additive = float(additive_info["range"][1])
                    default_additive = max(min_additive, 0.0)
                    st.session_state.input_values[ad] = st.number_input(
                        f"{additive_info['name']} 含量 ({unit_additive})",
                        min_value=min_additive, max_value=max_additive, value=default_additive, step=0.1, key=f"additive_{ad}"
                    )

        total = sum(st.session_state.input_values.values())
        is_only_pp = all(v == 0 for k, v in st.session_state.input_values.items() if k != "PP")

        with st.expander("✅ 输入验证"):
            if fraction_type in ["体积分数", "质量分数"]:
                if abs(total - 100.0) > 1e-6:
                    st.error(f"❗ {fraction_type}的总和必须为100%（当前：{total:.2f}%）")
                else:
                    st.success(f"{fraction_type}总和验证通过")
            else:
                st.success("成分总和验证通过")
                if is_only_pp:
                    st.info("检测到纯PP配方")
            selected_flame_keys = [key for key in flame_retardants if flame_retardants[key]["name"] in selected_flame_retardants]
            has_zinc_stannate = any("Zinc Stannate" in flame_retardants[key]["name"] or
                                   "Hydroxy Zinc Stannate" in flame_retardants[key]["name"]
                                   for key in selected_flame_keys)
            if not has_zinc_stannate:
                st.error("❗ 配方必须包含锡酸锌（Zinc Stannate）或羟基锡酸锌（Hydroxy Zinc Stannate）")
            else:
                st.success("配方验证通过，包含必要的阻燃成分")

        if st.button("🚀 开始预测", type="primary"):
            if fraction_type in ["体积分数", "质量分数"] and abs(total - 100.0) > 1e-6:
                st.error(f"预测中止：{fraction_type}的总和必须为100%")
                st.stop()
            if not has_zinc_stannate:
                st.error("预测中止：请添加锡酸锌或羟基锡酸锌")
                st.stop()

            if is_only_pp:
                loi_pred = 17.5
                ts_pred = 35.0
            else:
                if fraction_type == "体积分数":
                    total_vol = sum(st.session_state.input_values.values())
                    st.session_state.input_values = {
                        k: (v / total_vol * 100)
                        for k, v in st.session_state.input_values.items()
                    }
                all_features = sorted(
                    list(matrix_materials.keys()) +
                    list(flame_retardants.keys()) +
                    [key for category in additives for key in additives[category]]
                )
                loi_expected_features = 25
                ts_expected_features = 26
                loi_input_features = []
                ts_input_features = []
                for feature in all_features:
                    value = st.session_state.input_values.get(feature, 0.0)
                    loi_input_features.append(value)
                    ts_input_features.append(value)
                if len(loi_input_features) < loi_expected_features:
                    loi_input_features += [0.0] * (loi_expected_features - len(loi_input_features))
                elif len(loi_input_features) > loi_expected_features:
                    loi_input_features = loi_input_features[:loi_expected_features]
                if len(ts_input_features) < ts_expected_features:
                    ts_input_features += [0.0] * (ts_expected_features - len(ts_input_features))
                elif len(ts_input_features) > ts_expected_features:
                    ts_input_features = ts_input_features[:ts_expected_features]
                try:
                    loi_input = np.array([loi_input_features])
                    loi_scaled = models["loi_scaler"].transform(loi_input)
                    loi_pred = models["loi_model"].predict(loi_scaled)[0]
                    loi_pred = max(17.0, min(loi_pred, 50.0))
                except Exception as e:
                    st.error(f"LOI预测出错: {str(e)}")
                    loi_pred = 25.0
                try:
                    ts_input = np.array([ts_input_features])
                    ts_scaled = models["ts_scaler"].transform(ts_input)
                    ts_pred = models["ts_model"].predict(ts_scaled)[0]
                    ts_pred = max(0.0, min(ts_pred, 100.0))
                except Exception as e:
                    st.error(f"TS预测出错: {str(e)}")
                    ts_pred = 30.0

            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="LOI预测值", value=f"{loi_pred:.2f}%")
            with col2:
                st.metric(label="TS预测值", value=f"{ts_pred:.2f} MPa")

    elif page == "配方建议" and sub_page == "配方优化":
        render_inverse_design_page(models)

    elif page == "配方建议" and sub_page == "添加剂推荐":
        st.subheader("🧪 PVC添加剂智能推荐")
        try:
            predictor = Predictor("scaler_fold_1.pkl", "svc_fold_1.pkl")
        except Exception as e:
            st.error(f"模型加载失败: {str(e)}")
            predictor = None

        with st.expander("📋 参考样本", expanded=False):
            sample_data = [
                {"样本名称": "样本1", "推荐添加剂": "无添加剂",
                 "Sn%": 19.2, "添加比例": 0.0, "一甲%": 32.0,
                 "黄度值": [5.36, 6.29, 7.57, 8.57, 10.26, 13.21, 16.54, 27.47]},
                {"样本名称": "样本2", "推荐添加剂": "氯化石蜡",
                 "Sn%": 18.5, "添加比例": 3.64, "一甲%": 31.05,
                 "黄度值": [5.29, 6.83, 8.00, 9.32, 11.40, 14.12, 18.37, 30.29]},
                {"样本名称": "样本3", "推荐添加剂": "EA15（市售液体钙锌稳定剂）",
                 "Sn%": 19.0, "添加比例": 1.04, "一甲%": 31.88,
                 "黄度值": [5.24, 6.17, 7.11, 8.95, 10.33, 13.21, 17.48, 28.08]}
            ]
            for sample in sample_data:
                st.markdown(f"**{sample['样本名称']}** - {sample['推荐添加剂']}")
                cols = st.columns(4)
                cols[0].metric("Sn%", f"{sample['Sn%']}%")
                cols[1].metric("添加比例", f"{sample['添加比例']}%")
                cols[2].metric("一甲%", f"{sample['一甲%']}%")
                yellow_df = pd.DataFrame({
                    "时间(min)": [3, 6, 9, 12, 15, 18, 21, 24],
                    "黄度值": sample['黄度值']
                })
                st.dataframe(yellow_df.set_index("时间(min)"), use_container_width=True)

        if predictor:
            with st.form("additive_form"):
                st.subheader("参数输入")
                col1, col2, col3 = st.columns(3)
                with col1:
                    add_ratio = st.number_input("添加比例 (%)", min_value=0.0, max_value=100.0, value=3.64, step=0.1, format="%.2f")
                with col2:
                    sn_percent = st.number_input("Sn%", min_value=0.0, max_value=100.0, value=18.5, step=0.1, format="%.1f")
                with col3:
                    yijia_calculated = sn_percent / 0.6
                    st.markdown("**一甲含量（计算值）**")
                    st.markdown(f"`{yijia_calculated:.2f} %`")
                    st.caption("公式：一甲含量 = Sn含量 / 0.6")

                st.subheader("黄度值随时间变化（请尽可能提供足够多的时序黄度值，黄度值必须单调递增）")
                yellow_cols = st.columns(4)
                yellow_values = {}
                times = [3, 6, 9, 12, 15, 18, 21, 24]
                for i, time in enumerate(times):
                    with yellow_cols[i % 4]:
                        yellow_values[time] = st.number_input(
                            f"{time}min 黄度值", min_value=0.0, max_value=100.0,
                            value=5.29 + i * 3, step=0.1, format="%.2f", key=f"yellow_{time}"
                        )
                submit_btn = st.form_submit_button("🚀 生成推荐方案")

            if submit_btn:
                sample = [
                    sn_percent, add_ratio, yijia_calculated,
                    yellow_values[3], yellow_values[6], yellow_values[9], yellow_values[12],
                    yellow_values[15], yellow_values[18], yellow_values[21], yellow_values[24]
                ]
                try:
                    prediction = predictor.predict_one(sample)
                    result_map = {
                        1: "无推荐添加剂", 2: "氯化石蜡", 3: "EA12（脂肪酸复合醇酯）",
                        4: "EA15（液体钙锌稳定剂）", 5: "EA16（环氧化合物）",
                        6: "G70L（多官能团的脂肪酸复合酯混合物）", 7: "EA6（亚磷酸酯）"
                    }
                    additive_name = result_map.get(prediction, "未知类型")
                    additive_amount = add_ratio / 100
                    formula_data = [
                        ["PVC", 100.00], ["加工助剂ACR", 1.00], ["外滑剂70S", 0.35],
                        ["MBS", 5.00], ["316A", 0.20], ["稳定剂", 1.00]
                    ]
                    df = pd.DataFrame(formula_data, columns=["材料名称", "份数（基于PVC 100份）"])
                    st.success("添加剂推荐完成！")
                    col_res, col_table = st.columns([1, 2])
                    with col_res:
                        st.markdown(f"### **在添加剂比例为{additive_amount:.4f} 份时，推荐添加剂种类为**")
                        st.markdown(f"<div style='font-size:24px; color:#3f87a6; font-weight:bold; margin:10px 0;'>{additive_name}</div>", unsafe_allow_html=True)
                    with col_table:
                        st.markdown("### **完整配方表**")
                        st.dataframe(
                            df, use_container_width=True, hide_index=True,
                            column_config={
                                "材料名称": "材料名称",
                                "份数（基于PVC 100份）": st.column_config.NumberColumn("份数", format="%.4f")
                            }
                        )
                except Exception as e:
                    st.error(f"预测过程中出错: {str(e)}")
        else:
            st.warning("添加剂推荐功能暂时不可用，请检查模型文件是否存在")

    st.markdown("""
    <hr>
    <footer>
        <p>© 2025 阻燃聚合物复合材料智能设计平台</p>
        <p>声明：本平台仅供学术研究、技术验证等非营利性科研活动使用，严禁用于任何商业用途。</p>
    </footer>
    """, unsafe_allow_html=True)
else:
    show_homepage()
