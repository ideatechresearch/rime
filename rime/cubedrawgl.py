import os
import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *

import tkinter as tk
from tkinter import scrolledtext

from rime.cube import StickerCube, CubeBase
from rime.cubie import CubieBase
from rime.cubedraw import BaseCubeRenderer, RotationAnimation


class OpenGLCubeRenderer(BaseCubeRenderer):
    """OpenGL 渲染器，支持层旋转动画和交互控制"""

    COLOR_MAP = {
        'W': (1.0, 1.0, 1.0),
        'Y': (1.0, 0.84, 0.0),
        'R': (0.7, 0.0, 0.0),
        'O': (1.0, 0.4, 0.0),
        'G': (0.0, 0.55, 0.25),
        'B': (0.0, 0.3, 0.8),
    }

    WIDTH = 1000
    HEIGHT = 1000

    def __init__(self, cube, width=None, height=None):
        w = width or self.WIDTH
        h = height or self.HEIGHT
        super().__init__(cube, scale=(2.8 / cube.n))
        self.width = w
        self.height = h
        self._gl_ready = False

    def init_gl(self):
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width / self.height, 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -8.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.12, 0.12, 0.12, 1.0)

    def resize(self, width, height):
        self.width = width
        self.height = height
        if self._gl_ready:
            glViewport(0, 0, width, height)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45, width / height if height else 1.0, 0.1, 100.0)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glTranslatef(0.0, 0.0, -6.0)

    def zoom(self, delta):
        factor = 1.1 if delta > 0 else 0.9
        self.scale *= factor
        self.scale = max(self.scale, 0.05)
        self.scale = min(self.scale, 5.0)

    def draw(self):
        if not self._gl_ready:
            self.init_gl()
            self._gl_ready = True
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        glScalef(self.scale, self.scale, self.scale)
        glRotatef(np.degrees(self.angles[0]), 1, 0, 0)
        glRotatef(np.degrees(self.angles[1]), 0, 1, 0)
        glRotatef(np.degrees(self.angles[2]), 0, 0, 1)

        self.colors = self.cube.faces_colors
        for face, i, j, quad in self.compute_face_quads():
            q = np.array(quad)
            color = self.colors[face][i][j]
            is_ghost = False
            if self.partial is not None:
                axis, layer, ang = self.partial
                if CubeBase.should_rotate_by_sticker(face, i, j, axis, layer, self.n):
                    layer_geom = CubeBase.layer_to_geom(layer, self.n)
                    q = CubeBase.rotate_around_layer(q, axis, layer_geom, ang)
                    is_ghost = True
            self.draw_face_quad(q, color, ghost=is_ghost)
        glPopMatrix()

    @classmethod
    def draw_face_quad(cls, quad, color, ghost=False):
        r, g, b = cls.COLOR_MAP[color]
        ghost_alpha = 0.35 if ghost else 1.0
        glColor4f(r, g, b, ghost_alpha)
        glBegin(GL_QUADS)
        for x, y, z in quad:
            glVertex3f(x, y, z)
        glEnd()
        if ghost:
            glColor4f(0.8, 0.9, 1.0, 0.9)
            glLineWidth(2.5)
        else:
            glColor4f(0.1, 0.1, 0.1, 1.0)
            glLineWidth(1.5)
        glBegin(GL_LINE_LOOP)
        for x, y, z in quad:
            glVertex3f(x, y, z)
        glEnd()

    @classmethod
    def toggle_fullscreen(cls, is_fullscreen, width=None, height=None):
        w = width or cls.WIDTH
        h = height or cls.HEIGHT
        if is_fullscreen:
            info = pygame.display.Info()
            size = (info.current_w, info.current_h)
            flags = pygame.DOUBLEBUF | pygame.OPENGL | pygame.NOFRAME
        else:
            size = (w, h)
            flags = pygame.DOUBLEBUF | pygame.OPENGL
        screen = pygame.display.set_mode(size, flags)
        return screen, size


class OpenGLCubeApp:
    """OpenGL 魔方交互应用，支持动画队列和远程控制"""

    ROT_DURATION = 0.22

    def __init__(self, cube: StickerCube = None, n: int = 3):
        self.cube = cube or StickerCube(n=n)
        self.renderer = OpenGLCubeRenderer(self.cube)
        self.clock = pygame.time.Clock()
        self.pending = []
        self.current_anim = None
        self.paused = False
        self.auto_rotate = True
        self.view_drag = False
        self.last_mouse = (0, 0)
        self.face_dragging = False
        self.face_drag_start = None
        self.face_drag_face = None
        self.running = True
        self.fullscreen = False

        # 远程控制回调
        self.on_state_change = None

    def enqueue_moves(self, moves: list):
        if not isinstance(moves, (list, tuple)):
            moves = [moves]
        self.pending.extend(moves)

    def play(self):
        self.paused = False

    def pause(self):
        self.paused = True

    def update(self, dt: float):
        if self.auto_rotate and not self.view_drag:
            self.renderer.angles[1] += 0.2 * dt

        if self.paused:
            return

        if self.current_anim is None and self.pending:
            op = self.pending.pop(0)
            self.current_anim = RotationAnimation(*op, duration=self.ROT_DURATION)

        if self.current_anim is not None:
            done, angle = self.current_anim.step(dt)
            axis, layer, dir = self.current_anim.op
            self.renderer.apply_partial_rotation(axis, layer, np.radians(angle))
            if done:
                self.cube.rotate(axis, layer, dir)
                self.renderer.commit_partial()
                self.current_anim = None
                if self.on_state_change:
                    self.on_state_change(self.cube.get_state())

    def handle_mouse_down(self, pos, button):
        if button == 1:
            self.last_mouse = pos
            self.view_drag = True
        elif button == 3:
            self.face_dragging = True
            self.face_drag_start = pos
            self.face_drag_face = self._pick_face_at(pos)

    def handle_mouse_up(self, pos, button):
        if button == 1:
            self.view_drag = False
        elif button == 3:
            if self.face_dragging and self.face_drag_face is not None:
                dx = pos[0] - self.face_drag_start[0]
                dy = pos[1] - self.face_drag_start[1]
                axis, layer, direction = self._infer_turn_from_drag(self.face_drag_face, dx, dy)
                self.enqueue_moves([(axis, layer, direction)])
            self.face_dragging = False
            self.face_drag_face = None
            self.face_drag_start = None

    def handle_mouse_move(self, pos):
        if self.view_drag:
            dx = pos[0] - self.last_mouse[0]
            dy = pos[1] - self.last_mouse[1]
            self.renderer.angles[1] += dx * 0.005
            self.renderer.angles[0] += dy * 0.005
            self.last_mouse = pos

    def _pick_face_at(self, pos):
        cx, cy = self.renderer.width / 2, self.renderer.height / 2
        rx = (pos[0] - cx) / self.renderer.scale
        ry = (cy - pos[1]) / self.renderer.scale
        if ry > 1.2:
            return 'U'
        if ry < -1.2:
            return 'D'
        if rx > 1.2:
            return 'R'
        if rx < -1.2:
            return 'L'
        return 'F'

    def _infer_turn_from_drag(self, face, dx, dy):
        mid = self.cube.n // 2
        axis, side = CubeBase.face_axis[face]
        layer = -mid if side == 0 else mid
        major = dx if abs(dx) > abs(dy) else dy
        direction = 1 if major > 0 else -1
        return axis, layer, direction

    def run(self):
        pygame.init()
        pygame.display.set_mode(
            (self.renderer.width, self.renderer.height),
            pygame.DOUBLEBUF | pygame.OPENGL
        )
        pygame.display.set_caption("Rubik's Cube - OpenGL (Left drag=view, Right drag=turn)")

        while self.running:
            dt = self.clock.tick(60) / 1000.0
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.running = False
                elif ev.type == pygame.KEYDOWN:
                    self._handle_key(ev.key)
                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse_down(ev.pos, ev.button)
                elif ev.type == pygame.MOUSEBUTTONUP:
                    self.handle_mouse_up(ev.pos, ev.button)
                elif ev.type == pygame.MOUSEMOTION:
                    self.handle_mouse_move(ev.pos)
                elif ev.type == pygame.MOUSEWHEEL:
                    self.renderer.zoom(ev.y)

            self.update(dt)
            self.renderer.draw()
            pygame.display.flip()

        pygame.quit()

    def _handle_key(self, key):
        import random
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_f:
            self.fullscreen = not self.fullscreen
            screen, (w, h) = OpenGLCubeRenderer.toggle_fullscreen(
                self.fullscreen, self.renderer.width, self.renderer.height
            )
            self.renderer.resize(w, h)
        elif key == pygame.K_a:
            self.auto_rotate = not self.auto_rotate
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_p:
            seq = random.choice(self.cube.basic_generators())
            self.enqueue_moves([seq])
        elif key == pygame.K_g:
            moves = list(self.cube.generate_moves(25))
            self.enqueue_moves(moves)
        elif key == pygame.K_r:
            self.cube.reset()
        elif key == pygame.K_c:
            self.pending.clear()
        elif key == pygame.K_l:
            print(self.cube.faces_colors)
        elif key == pygame.K_n:
            self.cube.normalize()


class SocketBridge:
    """Socket 服务器，远程控制魔方"""

    def __init__(self, app: OpenGLCubeApp, host='127.0.0.1', port=9999):
        self.app = app
        self.host = host
        self.port = port
        self.running = False

    def start(self):
        import socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        self.running = True
        print(f"Socket server listening on {self.host}:{self.port}")

    def accept_and_handle(self):
        """接受连接并处理命令（非阻塞）"""
        import socket
        self.sock.setblocking(False)
        try:
            conn, addr = self.sock.accept()
            self._handle_client(conn, addr)
        except socket.error:
            pass

    def _handle_client(self, conn, addr):
        print(f"Connected from {addr}")
        try:
            data = conn.recv(1024).decode('utf-8')
            response = self._process_command(data)
            conn.sendall(response.encode('utf-8'))
        except Exception as e:
            print(f"Error: {e}")
        finally:
            conn.close()

    def _process_command(self, data: str) -> str:
        """处理命令并返回响应"""
        cmd = data.strip()
        if not cmd:
            return "ERROR: empty command"

        parts = cmd.split()
        action = parts[0].upper()

        try:
            if action == "MOVE":
                # MOVE axis layer direction (e.g., MOVE 0 1 1)
                if len(parts) != 4:
                    return "ERROR: MOVE requires axis layer direction"
                axis, layer, direction = int(parts[1]), int(parts[2]), int(parts[3])
                self.app.enqueue_moves([(axis, layer, direction)])
                return "OK"

            elif action == "MOVES":
                # MOVES axis1 layer1 dir1 axis2 layer2 dir2 ... (flattened)
                if len(parts) < 4 or (len(parts) - 1) % 3 != 0:
                    return "ERROR: MOVES requires multiples of 3 args"
                moves = []
                for i in range(1, len(parts), 3):
                    moves.append((int(parts[i]), int(parts[i + 1]), int(parts[i + 2])))
                self.app.enqueue_moves(moves)
                return "OK"

            elif action == "STATE":
                state = self.app.cube.get_state()
                import json
                return json.dumps(state.tolist()) if hasattr(state, 'tolist') else str(state)

            elif action == "RESET":
                self.app.cube.reset()
                return "OK"

            elif action == "SCRAMBLE":
                n = int(parts[1]) if len(parts) > 1 else 25
                moves = list(self.app.cube.generate_moves(n))
                self.app.enqueue_moves(moves)
                return f"OK {n} moves"

            elif action == "PAUSE":
                self.app.pause()
                return "OK"

            elif action == "PLAY":
                self.app.play()
                return "OK"

            elif action == "AUTO":
                self.app.auto_rotate = not self.app.auto_rotate
                return f"OK auto_rotate={self.app.auto_rotate}"

            elif action == "QUIT":
                self.app.running = False
                return "OK goodbye"

            else:
                return f"ERROR: unknown command '{action}'"

        except Exception as e:
            return f"ERROR: {e}"

    def stop(self):
        self.running = False
        if hasattr(self, 'sock'):
            self.sock.close()


def run_with_socket(cube=None, n=3, port=9999):
    """启动带 Socket 控制的 OpenGL 应用"""
    app = OpenGLCubeApp(cube, n)
    bridge = SocketBridge(app, port=port)

    pygame.init()
    pygame.display.set_mode(
        (app.renderer.width, app.renderer.height),
        pygame.DOUBLEBUF | pygame.OPENGL
    )
    pygame.display.set_caption("Rubik's Cube - OpenGL (+ socket control)")

    bridge.start()
    print("Press ESC to quit")

    while app.running:
        dt = app.clock.tick(60) / 1000.0

        # 非阻塞接受 socket 连接
        bridge.accept_and_handle()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                app.running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    app.running = False
                elif ev.key == pygame.K_f:
                    app.fullscreen = not app.fullscreen
                    screen, (w, h) = OpenGLCubeRenderer.toggle_fullscreen(
                        app.fullscreen, app.renderer.width, app.renderer.height
                    )
                    app.renderer.resize(w, h)
                elif ev.key == pygame.K_a:
                    app.auto_rotate = not app.auto_rotate
                elif ev.key == pygame.K_SPACE:
                    app.paused = not app.paused
                elif ev.key == pygame.K_p:
                    import random
                    seq = random.choice(app.cube.basic_generators())
                    app.enqueue_moves([seq])
                elif ev.key == pygame.K_g:
                    moves = list(app.cube.generate_moves(25))
                    app.enqueue_moves(moves)
                elif ev.key == pygame.K_r:
                    app.cube.reset()
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                app.handle_mouse_down(ev.pos, ev.button)
            elif ev.type == pygame.MOUSEBUTTONUP:
                app.handle_mouse_up(ev.pos, ev.button)
            elif ev.type == pygame.MOUSEMOTION:
                app.handle_mouse_move(ev.pos)
            elif ev.type == pygame.MOUSEWHEEL:
                app.renderer.zoom(ev.y)

        app.update(dt)
        app.renderer.draw()
        pygame.display.flip()

    pygame.quit()
    bridge.stop()


class TkinterCubeEditor:
    """Tkinter 2D 魔方贴纸编辑器，参考 client_gui.py"""

    # 使用与 CubeBase.FACES 一致的顺序: U, D, F, B, L, R
    FACES = CubeBase.FACES  # ['U', 'D', 'F', 'B', 'L', 'R']
    COLORS = CubeBase.COLORS
    COLS = ('white', 'green', 'red', 'yellow', 'blue', 'orange')
    COLOR_NAME_TO_IDX = {name: i for i, name in enumerate(COLS)}

    def __init__(self, root, cube: StickerCube = None, cell_size=50, on_state_change=None):
        self.cube = cube or StickerCube(n=3)
        self.cell_size = cell_size
        self.on_state_change = on_state_change

        # 创建 canvas（根据 FACE_OFFSET 计算尺寸）
        # U(1,0) D(1,2) -> width: (3+1+3)*cell_size, height: (1+3+1)*cell_size + some margin
        total_w = 12 * cell_size + 20
        total_h = 9 * cell_size + 20
        self.canvas = tk.Canvas(root, width=total_w, height=total_h)

        self.facelet_id = [[[0 for _ in range(3)] for _ in range(3)] for _ in range(6)]
        self.colorpick_id = [0] * 6
        self.curcol = self.COLS[0]  # 当前选中颜色

        self._create_facelets()
        self._create_colorpick()

        self.canvas.pack(side=tk.TOP)

    def _create_facelets(self):
        FACE_OFFSET = BaseCubeRenderer.FACE_OFFSET  # 展开图布局
        for f, face in enumerate(self.FACES):
            for row in range(3):
                y = 10 + FACE_OFFSET[face][1] * 3 * self.cell_size + row * self.cell_size
                for col in range(3):
                    x = 10 + FACE_OFFSET[face][0] * 3 * self.cell_size + col * self.cell_size
                    self.facelet_id[f][row][col] = self.canvas.create_rectangle(
                        x, y, x + self.cell_size, y + self.cell_size, fill='grey'
                    )
                    if row == 1 and col == 1:
                        self.canvas.create_text(
                            x + self.cell_size // 2, y + self.cell_size // 2,
                            font=('', 14), text=face, state=tk.DISABLED
                        )
        # 设置中心贴纸颜色（初始状态）
        for f, face in enumerate(self.FACES):
            self.canvas.itemconfig(self.facelet_id[f][1][1], fill=self.COLS[f])

    def _create_colorpick(self):
        for i, color in enumerate(self.COLS):
            x = (i % 3) * (self.cell_size + 5) + 7 * self.cell_size
            y = (i // 3) * (self.cell_size + 5) + 7 * self.cell_size
            self.colorpick_id[i] = self.canvas.create_rectangle(
                x, y, x + self.cell_size, y + self.cell_size, fill=color
            )
        self.canvas.itemconfig(self.colorpick_id[0], width=4)

    def _get_face_from_id(self, item_id):
        for f in range(6):
            for r in range(3):
                for c in range(3):
                    if self.facelet_id[f][r][c] == item_id:
                        return f, r, c
        return None, None, None

    def _on_click(self, event):
        item = self.canvas.find_withtag('current')
        if not item:
            return
        item = item[0]

        # 检查是否点击了颜色选择器
        for i, cid in enumerate(self.colorpick_id):
            if cid == item:
                self.curcol = self.COLS[i]
                for cid in self.colorpick_id:
                    self.canvas.itemconfig(cid, width=1)
                self.canvas.itemconfig(item, width=4)
                return

        # 检查是否点击了 facelet
        f, r, c = self._get_face_from_id(item)
        if f is not None:
            self.canvas.itemconfig(item, fill=self.curcol)
            self._sync_to_cube(f, r, c)

    def _tk_color_to_face_idx(self, tk_color: str) -> int:
        """tkinter 颜色名 -> CubeBase.FACES 索引"""
        idx = self.COLOR_NAME_TO_IDX.get(tk_color, -1)
        if idx < 0:
            return 0
        # COLS 顺序: white,green,red,yellow,blue,orange
        # 对应 COLORS: W,G,R,Y,B,O
        # 对应 FACES:  U,D,F,B,L,R  (solved 状态 f 面 = f)
        return idx

    def _sync_to_cube(self, f, r, c):
        """将 canvas 颜色同步到 cube 底层状态数组"""
        color = self.canvas.itemcget(self.facelet_id[f][r][c], 'fill')
        idx = self.COLOR_NAME_TO_IDX.get(color, 0)
        # 直接写底层 numpy 数组 cube.cube[f, r, c]
        self.cube.cube[f, r, c] = idx
        if self.on_state_change:
            self.on_state_change(self.cube.get_state())

    def sync_from_cube(self):
        """从 cube 底层状态数组同步到 canvas"""
        state = self.cube.cube  # (6, n, n) numpy array
        for f in range(6):
            for r in range(3):
                for c in range(3):
                    idx = int(state[f, r, c])
                    tk_color = self.COLS[idx]
                    self.canvas.itemconfig(self.facelet_id[f][r][c], fill=tk_color)

    def get_definition_string(self) -> str:
        """生成 54 字符的魔方定义串"""
        color_to_facelet = {}
        for i, face in enumerate(self.FACES):
            color_to_facelet[self.canvas.itemcget(self.facelet_id[i][1][1], 'fill')] = face

        s = ''
        for f in range(6):
            for row in range(3):
                for col in range(3):
                    color = self.canvas.itemcget(self.facelet_id[f][row][col], 'fill')
                    s += color_to_facelet.get(color, 'U')
        return s

    def clean(self):
        """恢复为干净魔方（中心色填充所有贴纸）"""
        self.cube.reset()
        self.sync_from_cube()

    def empty(self):
        """清空除中心外的所有贴纸（灰色=未定义，不同步到 cube）"""
        for f in range(6):
            for row in range(3):
                for col in range(3):
                    if row != 1 or col != 1:
                        self.canvas.itemconfig(self.facelet_id[f][row][col], fill='grey')

    def randomize(self):
        """随机打乱魔方并更新 canvas"""
        self.cube.reset()
        moves = list(self.cube.generate_moves(25))
        self.cube.apply(moves)
        self.sync_from_cube()

    def _sync_all_to_cube(self):
        for f, face in enumerate(self.FACES):
            for r in range(3):
                for c in range(3):
                    self._sync_to_cube(f, r, c)

    def bind_click(self):
        self.canvas.bind('<Button-1>', self._on_click)


def run_with_tkinter(cube=None, n=3, port=9999):
    """启动带 Tkinter UI（2D编辑 + 3D预览 + Socket控制）的 OpenGL 应用"""
    import threading

    cube = cube or StickerCube(n=n)
    app = OpenGLCubeApp(cube, n)
    bridge = SocketBridge(app, port=port)

    root = tk.Tk()
    root.title("Rubik's Cube - 2D Editor + 3D Preview")

    # ========== 左侧：2D 魔方编辑区 ==========
    editor = TkinterCubeEditor(root, cube=cube, cell_size=45,
                               on_state_change=lambda s: None)
    editor.bind_click()

    # ========== 右侧：控制面板 ==========
    right_panel = tk.Frame(root)
    right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)

    # 状态显示
    status_var = tk.StringVar(value="Ready")
    tk.Label(right_panel, textvariable=status_var, relief=tk.SUNKEN,
             bg='black', fg='lime', font=('Courier', 10), anchor=tk.W
             ).pack(fill=tk.X, pady=(0, 5))

    # 日志窗口
    log_text = scrolledtext.ScrolledText(right_panel, height=15, width=40,
                                         font=('Courier', 9))
    log_text.pack(fill=tk.BOTH, expand=True, pady=5)

    def log(msg):
        log_text.insert(tk.END, msg + '\n')
        log_text.see(tk.END)

    # ========== 功能按钮 ==========
    btn_row1 = tk.Frame(right_panel)
    btn_row1.pack(fill=tk.X, pady=2)
    tk.Button(btn_row1, text="Reset", command=lambda: (app.cube.reset(), editor.sync_from_cube(), log("Reset"))
              ).pack(side=tk.LEFT, fill=tk.X, expand=True)
    tk.Button(btn_row1, text="Scramble", command=lambda: (
        app.cube.reset(),
        moves := list(app.cube.generate_moves(25)),
        app.cube.apply(moves),
        editor.sync_from_cube(),
        log(f"Scramble: {len(moves)} moves")
    )).pack(side=tk.LEFT, fill=tk.X, expand=True)

    btn_row2 = tk.Frame(right_panel)
    btn_row2.pack(fill=tk.X, pady=2)
    tk.Button(btn_row2, text="Pause/Play", command=lambda: (
        setattr(app, 'paused', not app.paused),
        log("Paused" if app.paused else "Playing")
    )).pack(side=tk.LEFT, fill=tk.X, expand=True)
    tk.Button(btn_row2, text="Auto Rotate", command=lambda: (
        setattr(app, 'auto_rotate', not app.auto_rotate),
        log(f"Auto: {app.auto_rotate}")
    )).pack(side=tk.LEFT, fill=tk.X, expand=True)

    # ========== 求解区域 ==========
    solve_frame = tk.LabelFrame(right_panel, text="Solve")
    solve_frame.pack(fill=tk.X, pady=5)

    def cmd_solve():
        log("Solving...")
        root.update_idletasks()
        try:
            cubie_solver = CubieBase(n=n)
            moves = cubie_solver.solve_sticker(app.cube.get_state())
            log(f"Solution: {len(moves)} moves")
            if moves:
                app.enqueue_moves(moves)
        except Exception as e:
            log(f"Error: {e}")

    tk.Button(solve_frame, text="Solve (Local)", command=cmd_solve
              ).pack(fill=tk.X, pady=2)

    # ========== Socket 命令区 ==========
    cmd_frame = tk.Frame(right_panel)
    cmd_frame.pack(fill=tk.X, pady=5)

    cmd_entry = tk.Entry(cmd_frame, width=30)
    cmd_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def send_cmd():
        cmd = cmd_entry.get()
        if cmd:
            response = bridge._process_command(cmd)
            status_var.set(response)
            log(f"> {cmd}\n< {response}")
            cmd_entry.delete(0, tk.END)

    tk.Button(cmd_frame, text="Send", command=send_cmd).pack(side=tk.LEFT)

    # ========== 编辑器按钮 ==========
    edit_btn_row = tk.Frame(right_panel)
    edit_btn_row.pack(fill=tk.X, pady=5)
    tk.Button(edit_btn_row, text="Clean", command=lambda: (editor.clean(), log("Clean"))
              ).pack(side=tk.LEFT, fill=tk.X, expand=True)
    tk.Button(edit_btn_row, text="Empty", command=lambda: (editor.empty(), log("Empty"))
              ).pack(side=tk.LEFT, fill=tk.X, expand=True)
    tk.Button(edit_btn_row, text="Random", command=lambda: (editor.randomize(), log("Random"))
              ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    # Quit
    tk.Button(right_panel, text="Quit", command=lambda: (
        setattr(app, 'running', False),
        root.quit()
    ), bg='red', fg='white').pack(fill=tk.X, pady=10)

    # ========== 同步回调：canvas 变化时更新 3D ==========
    def on_editor_change(state):
        # 通知 OpenGL 渲染器更新
        pass

    editor.on_state_change = on_editor_change

    # ========== OpenGL 线程 ==========
    def opengl_loop():
        pygame.init()
        screen = pygame.display.set_mode(
            (app.renderer.width, app.renderer.height),
            pygame.DOUBLEBUF | pygame.OPENGL
        )
        pygame.display.set_caption("Rubik's Cube - OpenGL (Left drag=view, Right drag=turn)")

        bridge.start()
        log(f"Socket listening on port {port}")

        while app.running:
            dt = app.clock.tick(60) / 1000.0
            bridge.accept_and_handle()

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    app.running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        app.running = False
                    elif ev.key == pygame.K_f:
                        app.fullscreen = not app.fullscreen
                        screen, (w, h) = OpenGLCubeRenderer.toggle_fullscreen(
                            app.fullscreen, app.renderer.width, app.renderer.height
                        )
                        app.renderer.resize(w, h)
                    elif ev.key == pygame.K_a:
                        app.auto_rotate = not app.auto_rotate
                    elif ev.key == pygame.K_r:
                        app.cube.reset()
                    elif ev.key == pygame.K_s:
                        # 从当前状态求解
                        cubie_solver = CubieBase(n=n)
                        moves = cubie_solver.solve_sticker(app.cube.get_state())
                        log(f"Kociemba: {len(moves)} moves")
                        app.enqueue_moves(moves)
                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    app.handle_mouse_down(ev.pos, ev.button)
                elif ev.type == pygame.MOUSEBUTTONUP:
                    app.handle_mouse_up(ev.pos, ev.button)
                elif ev.type == pygame.MOUSEMOTION:
                    app.handle_mouse_move(ev.pos)
                elif ev.type == pygame.MOUSEWHEEL:
                    app.renderer.zoom(ev.y)

            app.update(dt)
            app.renderer.draw()
            pygame.display.flip()

        pygame.quit()
        bridge.stop()

    opengl_thread = threading.Thread(target=opengl_loop, daemon=True)
    opengl_thread.start()

    log("OpenGL thread started")
    root.mainloop()


if __name__ == "__main__":
    import sys

    # venv 下 Tcl/Tk 路径修复：venv 无法自动找到系统 Tcl 库
    if sys.platform == 'win32' and not os.environ.get('TCL_LIBRARY'):
        base = os.path.dirname(sys.executable)
        for candidate in [base, os.path.dirname(base)]:
            tcl_dir = os.path.join(candidate, 'tcl', 'tcl8.6')
            tk_dir = os.path.join(candidate, 'tcl', 'tk8.6')
            if os.path.isfile(os.path.join(tcl_dir, 'init.tcl')):
                os.environ['TCL_LIBRARY'] = tcl_dir
                os.environ['TK_LIBRARY'] = tk_dir
                break

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    # 方式1: 仅 OpenGL（保留原有功能）
    # OpenGLCubeRenderer.run(StickerCube(n=n))

    # 方式2: 带 Socket 控制台
    # run_with_socket(StickerCube(n=n), n=n, port=9999)

    # 方式3: 带 Tkinter UI + Socket
    run_with_tkinter(StickerCube(n=n), n=n, port=9999)
