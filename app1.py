import streamlit as st
import pandas as pd
import bcrypt
import os
from PIL import Image
import io
import base64
import joblib
import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer
import random

# --------------------- 初始化函数 ---------------------
def image_to_base64(image_path):
    """将图片转换为Base64编码，添加错误处理"""
    try:
        if os.path.exists(image_path):
            img = Image.open(image_path)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
        else:
            st.warning(f"图片文件不存在: {image_path}")
            # 创建一个简单的占位图片
            img = Image.new('RGB', (100, 100), color=(73, 109, 137))
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        st.error(f"图片加载失败: {e}")
        # 创建一个错误占位图片
        img = Image.new('RGB', (100, 100), color=(255, 0, 0))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

# --------------------- 全局配置 ---------------------
# 检查图片是否存在
icon_base64 = image_to_base64("图片1.jpg")
background_base64 = image_to_base64("BG.png")

# 设置页面配置
st.set_page_config(
    page_title="阻燃聚合物复合材料智能设计平台",
    layout="wide",
    page_icon=f"data:image/png;base64,{icon_base64}" if icon_base64 else None
)

# --------------------- 用户认证模块 ---------------------
USERS_FILE = "users.csv"
if not os.path.exists(USERS_FILE):
    pd.DataFrame(columns=["username", "password_hash", "email"]).to_csv(USERS_FILE, index=False)

def load_users():
    try:
        return pd.read_csv(USERS_FILE)
    except Exception as e:
        st.error(f"用户数据加载失败: {e}")
        return pd.DataFrame(columns=["username", "password_hash", "email"])

def save_user(username, password, email):
    users = load_users()
    if username in users['username'].values:
        return False
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    new_user = pd.DataFrame([[username, password_hash.decode(), email]],
                          columns=["username", "password_hash", "email"])
    users = pd.concat([users, new_user], ignore_index=True)
    try:
        users.to_csv(USERS_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"用户保存失败: {e}")
        return False

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
        try:
            users.to_csv(USERS_FILE, index=False)
            return True
        except Exception as e:
            st.error(f"密码重置失败: {e}")
            return False
    return False

# --------------------- 全局状态 ---------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = None

# --------------------- 样式配置 ---------------------
def apply_global_styles():
    """精准对齐样式方案+背景图"""
    # 确保背景图存在
    bg_base64 = background_base64 if background_base64 else ""
    
    st.markdown(f"""
    <style>
        /* 新增背景图设置 */
        .stApp {{
            position: relative;
            min-height: 100vh;
        }}
        .stApp::before {{
            content: "";
            background-image: url("data:image/png;base64,{bg_base64}");
            background-size: cover;
            background-position: center;
            opacity: 0.9;
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
        }}
        .stApp > div {{
            background-color: rgba(255, 255, 255, 0.85);
            min-height: 100vh;
        }}

        /* 保留原有对齐样式 */
        /* 父容器网格布局 */
        div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] {{
            display: grid !important;
            grid-template-columns: 1fr 1fr;
            gap: 20px !important;
            align-items: center !important;
            position: relative;  /* 新增层级控制 */
        }}

        /* 基体选择器统一样式 */
        #base-material-select {{
            height: 72px !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            background: rgba(255,255,255,0.9) !important;  /* 新增半透明背景 */
            border-radius: 8px !important;  /* 保持圆角一致 */
        }}
        #base-material-select [data-baseweb="select"] {{
            height: 72px !important;
            padding: 20px 24px !important;
            font-size: 18px !important;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            backdrop-filter: blur(2px);  /* 新增毛玻璃效果 */
        }}

        /* 含量输入框镜像样式 */
        div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"]:last-child {{
            height: 72px !important;
            padding: 20px 24px !important;
            font-size: 18px !important;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            display: flex !important;
            align-items: center !important;
            background: rgba(255,255,255,0.9) !important;  /* 调整为半透明 */
            backdrop-filter: blur(2px);  /* 新增效果 */
        }}

        /* 标签精准对齐增强 */
        label {{
            margin-bottom: 12px !important;
            font-size: 18px !important;
            transform: translateY(8px) !important;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);  /* 新增文字阴影 */
        }}

        /* 下拉菜单对齐修正 */
        [role="listbox"] {{
            margin-top: 8px !important;
            left: 0 !important;
            width: 100% !important;
            background: rgba(255,255,255,0.95) !important;  /* 半透明背景 */
            backdrop-filter: blur(4px);  /* 毛玻璃效果 */
        }}
        
        /* 侧边栏优化 */
        [data-testid="stSidebar"] {{
            background: rgba(255,255,255,0.92) !important;
            backdrop-filter: blur(6px);
        }}
        
        /* 主内容区域 */
        .main-content {{
            padding: 2rem;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-top: 1rem;
        }}
        
        /* 修复标题显示 */
        h1, h2, h3, h4, h5, h6 {{
            color: #1e3d59 !important;
        }}
    </style>
    """, unsafe_allow_html=True)

def render_global_header():
    """渲染全局头部组件"""
    # 确保图标存在
    icon = icon_base64 if icon_base64 else ""
    
    st.markdown(f"""
    <div class="global-header">
        <div style="max-width:1400px; margin:0 auto; display:flex; align-items:center; gap:3rem;">
            <img src="data:image/png;base64,{icon}" 
                 style="width:160px; height:auto; border-radius:16px; box-shadow:0 8px 32px rgba(0,0,0,0.2)"
                 alt="平台标志">
            <div>
                <h1 style="margin:0; font-size:3.5rem!important; color:#1e3d59!important;">
                    阻燃聚合物复合材料智能设计平台
                </h1>
                <p style="font-size:1.8rem!important; margin:1rem 0 0; color:#2c2c2c!important;">
                    Flame Retardant Composites AI Platform
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
# --------------------- 首页内容 ---------------------
def show_homepage():
    apply_global_styles()
    render_global_header()
    
    # 使用列布局确保内容正确显示
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div class="main-content">
            <div style="font-size:1.2rem; line-height:1.6; margin-bottom:2rem;">
                🚀 本平台融合AI与材料科学技术，致力于高分子复合材料的智能化设计，
                重点关注阻燃性能、力学性能和热稳定性的多目标优化与调控。
            </div>

            <h2 class="section-title">🌟 核心功能</h2>
            <div class="feature-card">
                <h3 style="font-size:1.4rem; color:var(--primary); margin:0 0 0.8rem 0;">
                    🔥 智能性能预测
                </h3>
                <p style="font-size:1.2rem;">
                    • 支持LOI（极限氧指数）预测<br>
                    • TS（拉伸强度）预测<br>
                </p>
            </div>

            <div class="feature-card">
                <h3 style="font-size:1.4rem; color:var(--primary); margin:0 0 0.8rem 0;">
                    ⚗️ 配方优化系统
                </h3>
                <p style="font-size:1.2rem;">
                    • 根据输入目标推荐配方<br>
                    • 支持选择配方种类<br>
                    • 添加剂比例智能推荐
                </p>
            </div>

            <h2 class="section-title">🏆 研究成果</h2>
            <div class="feature-card">
                <p style="font-size:1.2rem;">
                    Ma Weibin, Li Ling, Zhang Yu, et al.<br>
                    <em>Active learning-based generative design of halogen-free flame-retardant polymeric composites.</em><br>
                    <strong>Journal of Materials Informatics</strong> 2025;5:09.<br>
                    DOI: <a href="https://doi.org/10.20517/jmi.2025.09" target="_blank" 
                         style="color:var(--secondary); text-decoration:underline;">
                        10.20517/jmi.2025.09
                    </a>
                </p>
            </div>

            <div class="feature-card">
                <h2 class="section-title">👨💻 开发团队</h2>
                <p style="font-size:1.2rem;">
                    上海大学功能高分子<br>
                    PolyDesign <br>
                    马维宾 | 李凌 | 张瑜<br>
                    宋娜 | 丁鹏
                </p>
            </div>

            <div class="feature-card">
                <h2 class="section-title">🙏 项目支持</h2>
                <p style="font-size:1.2rem;">
                    云南省科技重点计划<br>
                    项目编号：202302AB080022<br>
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="auth-sidebar" style='
            background: rgba(255,255,255,0.98);
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        '>
        """, unsafe_allow_html=True)

        tab_login, tab_register, tab_forgot = st.tabs(["🔐 登录", "📝 注册", "🔑 忘记密码"])

        with tab_login:
            with st.form("login_form", clear_on_submit=True):
                st.markdown('<h3 style="text-align:center; margin-bottom:1.5rem;">用户登录</h3>', 
                          unsafe_allow_html=True)
                username = st.text_input("用户名", key="login_user")
                password = st.text_input("密码", type="password", key="login_pwd")
                
                if st.form_submit_button("立即登录", use_container_width=True):
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
                st.markdown('<h3 style="text-align:center; margin-bottom:1.5rem;">新用户注册</h3>', 
                          unsafe_allow_html=True)
                new_user = st.text_input("用户名（4-20位字母数字）", key="reg_user").strip()
                new_pwd = st.text_input("设置密码（至少6位）", type="password", key="reg_pwd")
                confirm_pwd = st.text_input("确认密码", type="password", key="reg_pwd_confirm")
                email = st.text_input("电子邮箱", key="reg_email")
                
                if st.form_submit_button("立即注册", use_container_width=True):
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
                st.markdown('<h3 style="text-align:center; margin-bottom:1.5rem;">密码重置</h3>', 
                          unsafe_allow_html=True)
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

        st.markdown('</div>', unsafe_allow_html=True)  # 结束auth-sidebar

# --------------------- 主流程控制 ---------------------
if not st.session_state.logged_in:
    show_homepage()
    st.stop()

# --------------------- 预测界面 ---------------------
if st.session_state.logged_in:
    # 应用样式和标题
    apply_global_styles()
    render_global_header()
    
    # 侧边栏主导航
    with st.sidebar:
        st.title("导航菜单")
        page = st.radio(
            "主功能选择",
            ["性能预测", "配方建议"],
            key="main_nav"
        )

        # 子功能选择（仅在配方建议时显示）
        sub_page = None
        if page == "配方建议":
            sub_page = st.radio(
                "子功能选择",
                ["配方优化", "添加剂推荐"],
                key="sub_nav"
            )
        
        if st.button("🚪 退出登录", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.success("已成功退出登录")
            st.rerun()

    # 显示当前页面标题
    st.subheader(f"🔮 {page}: {sub_page if sub_page else ''}")
    
    if page == "性能预测":
        # 在这里添加性能预测界面的内容
        st.info("性能预测功能正在开发中...")
        st.write("这是一个示例内容，实际功能将在后续版本中添加。")
        
    elif page == "配方建议":
        if sub_page == "配方优化":
            st.info("配方优化功能正在开发中...")
            st.write("这是一个示例内容，实际功能将在后续版本中添加。")
            
        elif sub_page == "添加剂推荐":
            st.info("添加剂推荐功能正在开发中...")
            st.write("这是一个示例内容，实际功能将在后续版本中添加。")

# 添加页脚
def add_footer():
    st.markdown("""
    <hr>
    <footer style="text-align: center; padding: 1rem; margin-top: 2rem;">
        <p>© 2025 阻燃聚合物复合材料智能设计平台</p>
        <p>声明：本平台仅供学术研究、技术验证等非营利性科研活动使用，严禁用于任何商业用途。</p>
    </footer>
    """, unsafe_allow_html=True)

add_footer()
