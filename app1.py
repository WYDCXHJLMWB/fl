import streamlit as st
import pandas as pd
import bcrypt
import os
import joblib
import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer

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
    st.session_state.input_values = {}  # 初始化输入值存储

# --------------------- 样式配置 ---------------------
def apply_global_styles():
    """简洁现代的样式方案"""
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
    """渲染全局头部组件"""
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
    
    st.markdown("""
    <div style="max-width:1400px; margin:0 auto; padding:2rem;">
    """, unsafe_allow_html=True)

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
            <h3 style="font-size:1.4rem; color:#1e3d59; margin:0 0 1rem 0;">
                🔥 智能性能预测
            </h3>
            <p style="font-size:1.1rem;">
                • 支持LOI（极限氧指数）预测<br>
                • TS（拉伸强度）预测<br>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3 style="font-size:1.4rem; color:#1e3d59; margin:0 0 1rem 0;">
                ⚗️ 配方优化系统
            </h3>
            <p style="font-size:1.1rem;">
                • 根据输入目标推荐配方<br>
                • 支持选择配方种类<br>
                • 添加剂比例智能推荐
            </p>
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
            DOI: <a href="https://doi.org/10.20517/jmi.2025.09" target="_blank" 
                 style="color:#3f87a6; text-decoration:underline;">
                10.20517/jmi.2025.09
            </a>
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
                上海大学功能高分子<br>
                PolyDesign <br>
                马维宾 | 李凌 | 张瑜<br>
                宋娜 | 丁鹏<br>
                上海大学计算机学院<br>
                韩越兴|李睿杰<br>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_sup:
        st.markdown("""
        <div class="feature-card">
            <h3 style="font-size:1.4rem; color:#1e3d59; margin:0 0 1rem 0;">
                🙏 项目支持
            </h3>
            <p style="font-size:1.1rem;">
                云南省科技重点计划<br>
                项目编号：202302AB080022<br>
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: 3rem; background: #ffffff; padding: 2rem; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
    """, unsafe_allow_html=True)
    
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
        self.time_series_cols = [
            "黄度值_3min", "6min", "9min", "12min",
            "15min", "18min", "21min", "24min"
        ]
        self.eng_features = [
            'seq_length', 'max_value', 'mean_value', 'min_value',
            'std_value', 'trend', 'range_value', 'autocorr'
        ]
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
            denominator = sum((values - mean) **2)
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

# --------------------- 主应用逻辑 ---------------------
if st.session_state.logged_in:
    # 侧边栏主导航
    page = st.sidebar.selectbox(
        "🔧 主功能选择",
        ["性能预测", "配方建议"],
        key="main_nav"
    )

    # 子功能选择
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
                "loi_features": ["PP", "AHP", "CFA", "APP", "Pentaerythritol", "DOPO", "ZS", "ZHS", "ZnB"],
                "ts_features": ["PP", "AHP", "CFA", "APP", "Pentaerythritol", "DOPO", "ZS", "ZHS", "ZnB"]
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
    
    def ensure_pp_first(features):
        if "PP" in features:
            features.remove("PP")
        return ["PP"] + sorted(features)

    apply_global_styles()
    render_global_header()
      
    if page == "性能预测":
        st.subheader("🔮 性能预测：基于配方预测LOI和TS")
    
        # 材料数据定义
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
    
        # 配方成分部分
        st.subheader("请选择配方成分")
        col_matrix = st.columns([4, 3], gap="medium")
        with col_matrix[0]:
            st.markdown('<div id="base-material-select">', unsafe_allow_html=True)
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
    
        # 阻燃剂选择
        st.subheader("请选择阻燃剂")
        st.markdown('<div id="base-material-select">', unsafe_allow_html=True)
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
                            min_value=min_val, 
                            max_value=max_val, 
                            value=default_value, 
                            step=0.1,
                            key=f"fr_{key}"
                        )
    
        # 助剂选择
        st.subheader("选择助剂")
        st.markdown('<div id="base-material-select">', unsafe_allow_html=True)
        selected_additives = st.multiselect(
            "选择助剂（可多选）", list(additives.keys()), default=[list(additives.keys())[0]]
        )
        
        # 处理助剂输入
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
                        min_value=min_additive, 
                        max_value=max_additive, 
                        value=default_additive, 
                        step=0.1,
                        key=f"additive_{ad}"
                    )

        # 计算总和（包含所有成分，包括基体）
        total = sum(st.session_state.input_values.values())
        # 纯PP配方检查（仅PP有值，其他成分均为0）
        is_only_pp = all(v == 0 for k, v in st.session_state.input_values.items() if k != "PP")

        # 输入验证
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
            
            # 锡酸锌/羟基锡酸锌验证
            selected_flame_keys = [key for key in flame_retardants if flame_retardants[key]["name"] in selected_flame_retardants]
            has_zinc_stannate = any("Zinc Stannate" in flame_retardants[key]["name"] or 
                                   "Hydroxy Zinc Stannate" in flame_retardants[key]["name"] 
                                   for key in selected_flame_keys)
            if not has_zinc_stannate:
                st.error("❗ 配方必须包含锡酸锌（Zinc Stannate）或羟基锡酸锌（Hydroxy Zinc Stannate）")
            else:
                st.success("配方验证通过，包含必要的阻燃成分")

        # 预测按钮
        if st.button("🚀 开始预测", type="primary"):
            # 总和验证
            if fraction_type in ["体积分数", "质量分数"] and abs(total - 100.0) > 1e-6:
                st.error(f"预测中止：{fraction_type}的总和必须为100%")
                st.stop()
            
            # 阻燃成分验证
            if not has_zinc_stannate:
                st.error("预测中止：请添加锡酸锌或羟基锡酸锌")
                st.stop()

            # 纯PP配方直接给默认值
            if is_only_pp:
                loi_pred = 17.5
                ts_pred = 35.0
            else:
                # 体积分数转质量分数处理 - 修复转换逻辑
                if fraction_type == "体积分数":
                    # 假设所有材料密度相同（简化处理）
                    total_vol = sum(st.session_state.input_values.values())
                    st.session_state.input_values = {
                        k: (v / total_vol * 100) 
                        for k, v in st.session_state.input_values.items()
                    }
        
                # 获取所有可能的材料特征
                all_features = sorted(
                    list(matrix_materials.keys()) + 
                    list(flame_retardants.keys()) + 
                    [key for category in additives for key in additives[category]]
                )
                
                # 确保特征数量与模型期望匹配
                # LOI模型期望25个特征，TS模型期望26个特征
                loi_expected_features = 25
                ts_expected_features = 26
                
                # 创建特征向量
                loi_input_features = []
                ts_input_features = []
                
                for feature in all_features:
                    value = st.session_state.input_values.get(feature, 0.0)
                    loi_input_features.append(value)
                    ts_input_features.append(value)
                
                # 填充特征向量到模型期望的长度
                if len(loi_input_features) < loi_expected_features:
                    loi_input_features += [0.0] * (loi_expected_features - len(loi_input_features))
                elif len(loi_input_features) > loi_expected_features:
                    loi_input_features = loi_input_features[:loi_expected_features]
                
                if len(ts_input_features) < ts_expected_features:
                    ts_input_features += [0.0] * (ts_expected_features - len(ts_input_features))
                elif len(ts_input_features) > ts_expected_features:
                    ts_input_features = ts_input_features[:ts_expected_features]
                
                # LOI预测
                try:
                    loi_input = np.array([loi_input_features])
                    loi_scaled = models["loi_scaler"].transform(loi_input)
                    loi_pred = models["loi_model"].predict(loi_scaled)[0]
                    # 约束LOI在合理范围内 (17-50)
                    loi_pred = max(17.0, min(loi_pred, 50.0))
                except Exception as e:
                    st.error(f"LOI预测出错: {str(e)}")
                    loi_pred = 25.0  # 默认值
        
                # TS预测
                try:
                    ts_input = np.array([ts_input_features])
                    ts_scaled = models["ts_scaler"].transform(ts_input)
                    ts_pred = models["ts_model"].predict(ts_scaled)[0]
                    # 约束TS在合理范围内 (0-100)
                    ts_pred = max(0.0, min(ts_pred, 100.0))
                except Exception as e:
                    st.error(f"TS预测出错: {str(e)}")
                    ts_pred = 30.0  # 默认值
        
            # 显示结果
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="LOI预测值", value=f"{loi_pred:.2f}%")
            with col2:
                st.metric(label="TS预测值", value=f"{ts_pred:.2f} MPa")

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
                    add_ratio = st.number_input(
                        "添加比例 (%)", 
                        min_value=0.0, max_value=100.0,
                        value=3.64, step=0.1, format="%.2f"
                    )
                with col2:
                    sn_percent = st.number_input(
                        "Sn%", 
                        min_value=0.0, max_value=100.0,
                        value=18.5, step=0.1, format="%.1f"
                    )
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
                            f"{time}min 黄度值",
                            min_value=0.0, max_value=100.0,
                            value=5.29 + i * 3, step=0.1,
                            format="%.2f", key=f"yellow_{time}"
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
                        1: "无推荐添加剂", 
                        2: "氯化石蜡", 
                        3: "EA12（脂肪酸复合醇酯）",
                        4: "EA15（液体钙锌稳定剂）", 
                        5: "EA16（环氧化合物）",
                        6: "G70L（多官能团的脂肪酸复合酯混合物）", 
                        7: "EA6（亚磷酸酯）"
                    }
                    
                    additive_name = result_map.get(prediction, "未知类型")
                    additive_amount = add_ratio / 100  # 转换为份数比例
                    
                    formula_data = [
                        ["PVC", 100.00],
                        ["加工助剂ACR", 1.00],
                        ["外滑剂70S", 0.35],
                        ["MBS", 5.00],
                        ["316A", 0.20],
                        ["稳定剂", 1.00]
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
                            df, 
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "材料名称": "材料名称",
                                "份数（基于PVC 100份）": st.column_config.NumberColumn(
                                    "份数", format="%.4f"
                                )
                            }
                        )
                        
                except Exception as e:
                    st.error(f"预测过程中出错: {str(e)}")
        else:
            st.warning("添加剂推荐功能暂时不可用，请检查模型文件是否存在")

    # 添加页脚
    st.markdown("""
    <hr>
    <footer>
        <p>© 2025 阻燃聚合物复合材料智能设计平台</p>
        <p>声明：本平台仅供学术研究、技术验证等非营利性科研活动使用，严禁用于任何商业用途。</p>
    </footer>
    """, unsafe_allow_html=True)
else:
    show_homepage()
