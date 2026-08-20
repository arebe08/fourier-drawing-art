
import math
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="푸리에 드로잉 체험",
    page_icon="🌀",
    layout="wide",
)

# -----------------------------
# 1. 목표 그림 생성
# -----------------------------
def resample_closed_polyline(points, n=1024):
    """폐곡선 꼭짓점들을 호길이에 가깝게 균등 재표본화한다."""
    pts = np.asarray(points, dtype=float)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    seg = np.diff(pts, axis=0)
    seg_len = np.sqrt((seg**2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]

    s_new = np.linspace(0, total, n, endpoint=False)
    x_new = np.interp(s_new, cum, pts[:, 0])
    y_new = np.interp(s_new, cum, pts[:, 1])
    return x_new + 1j * y_new


def normalize_path(z):
    z = z - np.mean(z)
    scale = np.max(np.abs(z))
    if scale == 0:
        return z
    return z / scale


def make_cat(n=1024):
    # 한 줄로 따라갈 수 있는 고양이 얼굴 외곽선
    pts = [
        (-0.95, -0.20), (-0.92, 0.32), (-0.82, 0.98),
        (-0.36, 0.70), (-0.12, 0.78), (0.12, 0.78),
        (0.36, 0.70), (0.82, 0.98), (0.92, 0.32),
        (0.95, -0.20), (0.82, -0.58), (0.56, -0.82),
        (0.25, -0.96), (0.00, -1.02), (-0.25, -0.96),
        (-0.56, -0.82), (-0.82, -0.58),
    ]
    return normalize_path(resample_closed_polyline(pts, n))


def make_fish(n=1024):
    # 몸통과 꼬리가 한 획으로 이어지는 물고기 실루엣
    pts = [
        (-0.95, 0.00), (-0.64, 0.34), (-0.20, 0.55),
        (0.30, 0.50), (0.72, 0.26), (0.98, 0.00),
        (0.72, -0.26), (0.30, -0.50), (-0.20, -0.55),
        (-0.64, -0.34), (-0.95, 0.00),
        (-1.35, -0.52), (-1.25, 0.00), (-1.35, 0.52),
        (-0.95, 0.00),
    ]
    return normalize_path(resample_closed_polyline(pts, n))


def make_butterfly(n=1024):
    # 좌우 날개와 몸체가 연결된 단순화 나비 외곽선
    pts = [
        (0.00, 0.72),
        (-0.18, 0.82), (-0.48, 0.98), (-0.88, 0.88),
        (-1.05, 0.58), (-0.93, 0.25), (-0.68, 0.08),
        (-0.92, -0.18), (-0.86, -0.55), (-0.55, -0.72),
        (-0.22, -0.42), (0.00, -0.12),
        (0.22, -0.42), (0.55, -0.72), (0.86, -0.55),
        (0.92, -0.18), (0.68, 0.08), (0.93, 0.25),
        (1.05, 0.58), (0.88, 0.88), (0.48, 0.98),
        (0.18, 0.82),
    ]
    return normalize_path(resample_closed_polyline(pts, n))


def make_flower(n=1024):
    # 꽃잎의 크기를 조금씩 달리해 완전히 단순한 장미곡선보다 복잡하게 구성
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = (
        0.72
        + 0.18 * np.cos(5 * theta)
        + 0.08 * np.sin(3 * theta + 0.6)
        + 0.05 * np.cos(8 * theta - 0.4)
    )
    z = r * np.exp(1j * theta)
    return normalize_path(z)


SHAPES = {
    "고양이 얼굴": {"maker": make_cat, "circles": 10, "difficulty": "중급"},
    "물고기": {"maker": make_fish, "circles": 10, "difficulty": "중급"},
    "나비": {"maker": make_butterfly, "circles": 12, "difficulty": "상급"},
    "꽃 패턴": {"maker": make_flower, "circles": 12, "difficulty": "상급"},
}


# -----------------------------
# 2. 푸리에 계수 계산
# -----------------------------
def fourier_components(z, n_circles):
    """
    z(t)를 복소수 평면의 폐곡선으로 보고 DFT 계수를 계산한다.
    DC(중심 이동) 성분을 제외한 뒤 진폭이 큰 성분부터 n개 선택한다.
    """
    n = len(z)
    coeff = np.fft.fft(z) / n
    freqs = np.fft.fftfreq(n, d=1 / n).astype(int)

    components = []
    dc = 0j
    for c, f in zip(coeff, freqs):
        if f == 0:
            dc = c
        else:
            components.append(
                {
                    "freq": int(f),
                    "radius": float(abs(c)),
                    "phase": float(np.angle(c)),
                    "coef": complex(c),
                }
            )

    components.sort(key=lambda item: item["radius"], reverse=True)
    return dc, components[:n_circles]


def reconstruct(components, radii=None, samples=720, dc=0j):
    t = np.linspace(0, 1, samples, endpoint=False)
    z = np.full(samples, dc, dtype=complex)

    if radii is None:
        radii = [c["radius"] for c in components]

    for c, r in zip(components, radii):
        z += r * np.exp(1j * (2 * np.pi * c["freq"] * t + c["phase"]))
    return t, z


def similarity_score(user_z, target_z):
    """
    같은 주파수/위상을 사용한 두 궤적의 정규화 RMSE를 0~100점으로 변환.
    완전히 일치하면 100점.
    """
    rmse = np.sqrt(np.mean(np.abs(user_z - target_z) ** 2))
    scale = np.sqrt(np.mean(np.abs(target_z) ** 2))
    if scale < 1e-12:
        return 100.0
    nrmse = rmse / scale
    return float(np.clip(100 * (1 - nrmse), 0, 100))


def direction_hint(user_r, target_r):
    tol = max(0.012, target_r * 0.07)
    diff = user_r - target_r
    if abs(diff) <= tol:
        return "적절함"
    if diff < 0:
        return "더 크게"
    return "더 작게"


# -----------------------------
# 3. 그래프
# -----------------------------
def path_figure(target, user=None, title="드로잉 결과"):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=target.real,
            y=target.imag,
            mode="lines",
            name="목표",
            line=dict(width=3),
        )
    )

    if user is not None:
        fig.add_trace(
            go.Scatter(
                x=user.real,
                y=user.imag,
                mode="lines",
                name="학생 결과",
                line=dict(width=3, dash="dash"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=45, b=10),
        height=480,
        legend=dict(orientation="h"),
    )
    return fig


def epicycle_animation(components, radii, dc=0j, frame_count=72):
    """
    연결된 원(에피사이클), 반지름 벡터, 끝점의 누적 궤적을 Plotly 애니메이션으로 만든다.
    """
    times = np.linspace(0, 1, frame_count, endpoint=False)
    trail = []

    def frame_data(t):
        current = complex(dc)
        arm_x = [current.real]
        arm_y = [current.imag]
        circle_x, circle_y = [], []

        for comp, radius in zip(components, radii):
            ang = 2 * np.pi * comp["freq"] * t + comp["phase"]

            phi = np.linspace(0, 2 * np.pi, 42)
            circle_x.extend((current.real + radius * np.cos(phi)).tolist())
            circle_y.extend((current.imag + radius * np.sin(phi)).tolist())
            circle_x.append(None)
            circle_y.append(None)

            current = current + radius * np.exp(1j * ang)
            arm_x.append(current.real)
            arm_y.append(current.imag)

        trail.append(current)

        return [
            go.Scatter(
                x=circle_x, y=circle_y, mode="lines",
                name="회전 원", line=dict(width=1), hoverinfo="skip"
            ),
            go.Scatter(
                x=arm_x, y=arm_y, mode="lines+markers",
                name="반지름", line=dict(width=2), marker=dict(size=4),
                hoverinfo="skip"
            ),
            go.Scatter(
                x=[p.real for p in trail], y=[p.imag for p in trail],
                mode="lines", name="그려진 궤적", line=dict(width=3),
                hoverinfo="skip"
            ),
            go.Scatter(
                x=[current.real], y=[current.imag],
                mode="markers", name="끝점", marker=dict(size=8),
                hoverinfo="skip"
            ),
        ]

    all_frame_data = []
    trail.clear()
    for idx, t in enumerate(times):
        all_frame_data.append(
            go.Frame(data=frame_data(t), name=str(idx))
        )

    # 마지막 구간까지 자연스럽게 보이도록 첫 프레임을 초기 데이터로 사용
    initial = all_frame_data[0].data

    total_r = sum(radii) + abs(dc)
    lim = max(1.25, total_r * 1.05)

    fig = go.Figure(data=initial, frames=all_frame_data)
    fig.update_layout(
        title="에피사이클 애니메이션",
        xaxis=dict(range=[-lim, lim], visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-lim, lim], visible=False),
        height=560,
        margin=dict(l=10, r=10, t=45, b=10),
        showlegend=False,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.02,
                y=1.02,
                buttons=[
                    dict(
                        label="▶ 재생",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 80, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                    dict(
                        label="⏸ 일시정지",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
    )
    return fig


# -----------------------------
# 4. Streamlit UI
# -----------------------------
st.title("푸리에 드로잉: 원의 크기를 맞혀 그림을 완성해 보자")
st.write(
    "복잡한 윤곽선은 서로 다른 속도와 시작각을 가진 여러 회전 운동의 합으로 "
    "근사할 수 있습니다. 이 활동에서는 각 원의 **회전속도(주파수)**와 "
    "**시작각(위상)**은 프로그램이 고정하고, 여러분이 **원의 반지름(진폭)**을 "
    "직접 추정합니다."
)

with st.expander("활동 방법 보기", expanded=False):
    st.markdown(
        """
1. 목표 그림을 하나 선택합니다.
2. 프로그램이 사용할 원의 개수를 알려줍니다.
3. 각 원의 반지름을 직접 입력합니다.
4. **결과 확인**을 눌러 목표 그림과의 유사도를 확인합니다.
5. 각 원에 대한 방향 힌트(더 크게 / 더 작게 / 적절함)를 보고 수정합니다.
6. 원하는 정확도에 도달할 때까지 반복합니다.
7. 완성 후 에피사이클 애니메이션으로 여러 회전 운동이 하나의 그림을 만드는 과정을 관찰합니다.
        """
    )

shape_name = st.selectbox("1. 목표 그림 선택", list(SHAPES.keys()))
shape_info = SHAPES[shape_name]
n_circles = shape_info["circles"]

raw_target = shape_info["maker"](1024)
dc, components = fourier_components(raw_target, n_circles)
_, target_n = reconstruct(components, dc=dc, samples=720)

left, right = st.columns([1, 1])

with left:
    st.subheader(f"목표: {shape_name}")
    st.caption(f"난이도: {shape_info['difficulty']} / 사용 원: {n_circles}개")
    st.plotly_chart(
        path_figure(raw_target, title="원본 목표 윤곽"),
        use_container_width=True,
        config={"displayModeBar": False},
    )

with right:
    st.subheader("이 활동의 목표")
    st.write(
        f"이 그림은 프로그램에서 **{n_circles}개의 주요 푸리에 성분**을 사용해 근사합니다. "
        "정답 반지름은 공개하지 않습니다."
    )
    target_accuracy = st.slider(
        "내가 목표로 할 정확도",
        min_value=70,
        max_value=98,
        value=90,
        step=1,
        key=f"accuracy_{shape_name}",
    )
    st.info(
        "원의 번호는 진폭이 큰 주요 성분부터 정렬되어 있습니다. "
        "큰 번호로 갈수록 대체로 더 세밀한 윤곽 보정에 관여합니다."
    )

# 그림별 세션 상태
history_key = f"history_{shape_name}"
attempt_key = f"attempt_{shape_name}"
last_key = f"last_result_{shape_name}"

if history_key not in st.session_state:
    st.session_state[history_key] = []
if attempt_key not in st.session_state:
    st.session_state[attempt_key] = 0
if last_key not in st.session_state:
    st.session_state[last_key] = None

st.divider()
st.subheader("2. 각 원의 반지름을 추정해 입력하세요")

largest = max(c["radius"] for c in components)
max_input = max(1.2, largest * 1.8)

with st.form(key=f"radius_form_{shape_name}"):
    cols = st.columns(2)
    user_radii = []

    for i, comp in enumerate(components):
        # 모든 값을 똑같이 주지 않고, 정답을 노출하지 않는 작은 기본값을 사용한다.
        default = min(0.12, max_input / 5)
        with cols[i % 2]:
            r = st.number_input(
                f"{i+1}번 원 반지름",
                min_value=0.0,
                max_value=float(max_input),
                value=float(default),
                step=0.01,
                format="%.2f",
                key=f"r_{shape_name}_{i}",
            )
            st.caption(
                f"회전속도: {comp['freq']:+d}회/주기 · 시작각은 프로그램에서 고정"
            )
            user_radii.append(float(r))

    submitted = st.form_submit_button("결과 확인", type="primary", use_container_width=True)

if submitted:
    _, user_z = reconstruct(components, radii=user_radii, dc=dc, samples=720)
    score = similarity_score(user_z, target_n)

    st.session_state[attempt_key] += 1
    st.session_state[history_key].append(score)
    st.session_state[last_key] = {
        "radii": user_radii,
        "score": score,
        "user_z_real": user_z.real.tolist(),
        "user_z_imag": user_z.imag.tolist(),
    }

last = st.session_state[last_key]

if last is not None:
    user_radii = list(last["radii"])
    user_z = np.array(last["user_z_real"]) + 1j * np.array(last["user_z_imag"])
    score = float(last["score"])

    st.divider()
    st.subheader("3. 결과와 피드백")

    c1, c2, c3 = st.columns(3)
    c1.metric("현재 유사도", f"{score:.1f}%")
    c2.metric("목표 정확도", f"{target_accuracy}%")
    c3.metric("시도 횟수", f"{st.session_state[attempt_key]}회")

    if score >= target_accuracy:
        st.success(
            f"목표 정확도 {target_accuracy}%에 도달했습니다. "
            "이제 아래 애니메이션에서 여러 회전 운동이 합쳐지는 과정을 확인해 보세요."
        )
    else:
        st.warning(
            f"목표까지 {target_accuracy - score:.1f}%p 남았습니다. "
            "아래 방향 힌트를 참고해 다시 조정해 보세요."
        )

    compare_col, hint_col = st.columns([1.25, 1])

    with compare_col:
        st.plotly_chart(
            path_figure(target_n, user_z, title="목표 근사와 학생 결과 비교"),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with hint_col:
        st.markdown("#### 원별 방향 힌트")
        st.caption("정답 숫자는 보여주지 않고 조절 방향만 알려줍니다.")
        for i, (user_r, comp) in enumerate(zip(user_radii, components)):
            hint = direction_hint(user_r, comp["radius"])
            symbol = "✓" if hint == "적절함" else ("↑" if hint == "더 크게" else "↓")
            st.write(f"**{i+1}번 원**: {symbol} {hint}")

    if len(st.session_state[history_key]) >= 2:
        st.markdown("#### 시행착오 기록")
        attempts = list(range(1, len(st.session_state[history_key]) + 1))
        hist_fig = go.Figure()
        hist_fig.add_trace(
            go.Scatter(
                x=attempts,
                y=st.session_state[history_key],
                mode="lines+markers",
                name="유사도",
            )
        )
        hist_fig.add_hline(y=target_accuracy, line_dash="dash")
        hist_fig.update_layout(
            xaxis_title="시도 횟수",
            yaxis_title="유사도(%)",
            yaxis_range=[0, 100],
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(hist_fig, use_container_width=True)

    st.markdown("#### 에피사이클로 실제 그려보기")
    st.write(
        "각 원의 중심이 앞 원의 끝점에 붙어 회전하고, 마지막 원의 끝점이 "
        "그리는 궤적이 최종 드로잉이 됩니다."
    )
    st.plotly_chart(
        epicycle_animation(components, user_radii, dc=dc),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    with st.expander("심화 관찰: 위상과 원의 개수는 왜 중요한가?"):
        st.markdown(
            """
- **반지름(진폭)**: 해당 회전 성분이 그림에 얼마나 크게 기여하는지를 나타냅니다.
- **회전속도(주파수)**: 한 주기 동안 해당 원이 몇 번 회전하는지를 나타냅니다.
- **시작각(위상)**: 회전이 어디에서 시작하는지를 나타냅니다.
- 반지름이 같아도 위상을 바꾸면 성분들이 서로 다른 위치에서 더해지므로 전체 그림이 달라질 수 있습니다.
- 원의 개수를 늘리면 일반적으로 더 세밀한 고주파 성분까지 포함할 수 있어 원본 윤곽을 더 정교하게 근사할 수 있습니다.
            """
        )

    # 위상 교란 체험
    if st.checkbox("위상을 무작위로 바꾼 결과도 비교해 보기", key=f"phase_demo_{shape_name}"):
        rng = np.random.default_rng(2026)
        changed = []
        for comp in components:
            c = dict(comp)
            c["phase"] = float(rng.uniform(-np.pi, np.pi))
            changed.append(c)
        _, phase_z = reconstruct(changed, radii=user_radii, dc=dc, samples=720)

        st.plotly_chart(
            path_figure(user_z, phase_z, title="같은 반지름, 다른 위상 비교"),
            use_container_width=True,
            config={"displayModeBar": False},
        )

if st.button("현재 그림의 시도 기록 초기화", key=f"reset_{shape_name}"):
    st.session_state[history_key] = []
    st.session_state[attempt_key] = 0
    st.session_state[last_key] = None
    st.rerun()

st.divider()
st.caption(
    "학습용 프로그램: 목표 그림의 푸리에 주파수·위상은 미리 계산하고, "
    "학생은 진폭을 조절하여 복잡한 형태가 여러 주기 성분의 합으로 구성되는 과정을 체험합니다."
)
