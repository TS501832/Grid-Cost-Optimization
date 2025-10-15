import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba
import heapq
import random
import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    pass

# =======================
# CONFIG
# =======================
GRID_W, GRID_H = 40, 30
OBSTACLE_COUNT = 60
POINT_COUNT = 5
INTERVAL_MS = 25

# =======================
# GRID & TERRAIN
# =======================
grid = np.zeros((GRID_H, GRID_W), dtype=int)
for _ in range(OBSTACLE_COUNT):
    x = random.randint(0, GRID_W - 1)
    y = random.randint(0, GRID_H - 1)
    grid[y, x] = 1  # wall

terrain_types = {
    0: ("#b5e7a0", 1, 0.35),
    1: ("#d2b48c", 2, 0.35),
    2: ("#d3d3d3", 3, 0.35),
}
terrain_map = np.full_like(grid, -1, dtype=int)
for y in range(GRID_H):
    for x in range(GRID_W):
        if grid[y, x] == 0:
            terrain_map[y, x] = random.randint(0, 2)

# =======================
# POINTS
# =======================
points = []
while len(points) < POINT_COUNT:
    x = random.randint(0, GRID_W - 1)
    y = random.randint(0, GRID_H - 1)
    if grid[y, x] == 0:
        points.append((x, y))

start_blue, start_green = points[0], points[1]
other_points = points[2:]

# =======================
# RANDOM GOAL LINE
# =======================
def in_bounds(x, y):
    return 0 <= x < GRID_W and 0 <= y < GRID_H

def generate_random_goal_line():
    orientation = random.choice(["horizontal", "vertical", "diagonal"])
    if orientation == "horizontal":
        y = random.randint(5, GRID_H - 6)
        x1, x2 = sorted(random.sample(range(3, GRID_W - 3), 2))
        line = [(x, y) for x in range(x1, x2 + 1) if grid[y, x] == 0]
    elif orientation == "vertical":
        x = random.randint(5, GRID_W - 6)
        y1, y2 = sorted(random.sample(range(3, GRID_H - 3), 2))
        line = [(x, y) for y in range(y1, y2 + 1) if grid[y, x] == 0]
    else:  # diagonal
        x1, y1 = random.randint(3, GRID_W - 6), random.randint(3, GRID_H - 6)
        length = random.randint(5, min(GRID_W, GRID_H) // 2)
        line = []
        for i in range(length):
            x, y = x1 + i, y1 + i
            if in_bounds(x, y) and grid[y, x] == 0:
                line.append((x, y))
    return line

goal_line = generate_random_goal_line()

# =======================
# HELPERS
# =======================
DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
def is_free(x,y): return in_bounds(x,y) and grid[y,x]==0
def can_move(ax,ay,bx,by):
    if not is_free(bx,by): return False
    dx,dy=bx-ax,by-ay
    if abs(dx)<=1 and abs(dy)<=1:
        if dx!=0 and dy!=0:
            if not (is_free(ax+dx,ay) and is_free(ax,ay+dy)): return False
        return True
    return False
def terrain_cost(x,y):
    if not is_free(x,y): return np.inf
    _,czk,_=terrain_types[terrain_map[y,x]]
    return int(czk)

# =======================
# DIJKSTRA COST-BASED
# =======================
def dijkstra_path(start, goals):
    H,W = grid.shape
    dist = np.full((H,W), np.iinfo(np.int64).max, dtype=np.int64)
    parent = np.full((H,W,2), -1, dtype=int)
    sx,sy=start
    pq=[]
    dist[sy,sx]=0
    heapq.heappush(pq,(0,sx,sy))
    goal_set=set(goals)
    best_goal=None
    while pq:
        d,x,y=heapq.heappop(pq)
        if d!=dist[y,x]: continue
        if (x,y) in goal_set:
            best_goal=(x,y)
            path=[]
            cur=(x,y)
            while cur!=(-1,-1):
                path.append(cur)
                px,py=parent[cur[1],cur[0]]
                if px==-1: break
                cur=(px,py)
            return list(reversed(path)), best_goal
        for (dx,dy) in DIRS:
            nx,ny=x+dx,y+dy
            if not can_move(x,y,nx,ny): continue
            nd=d+terrain_cost(nx,ny)
            if nd<dist[ny,nx]:
                dist[ny,nx]=nd
                parent[ny,nx]=(x,y)
                heapq.heappush(pq,(nd,nx,ny))
    return [], None

# =======================
# PATHS TO GOAL LINE
# =======================
path_blue, blue_goal_pt = dijkstra_path(start_blue, goal_line)
path_green, green_goal_pt = dijkstra_path(start_green, goal_line)

# =======================
# COST FUNCTION
# =======================
def path_cost(p): 
    return sum(terrain_cost(x,y) for (x,y) in p if np.isfinite(terrain_cost(x,y)))

# =======================
# SKELETON (meeting + joint path)
# =======================
def full_dijkstra_cost_map(start):
    H, W = grid.shape
    dist = np.full((H, W), np.iinfo(np.int64).max, dtype=np.int64)
    sx, sy = start
    pq = [(0, sx, sy)]
    dist[sy, sx] = 0
    while pq:
        d, x, y = heapq.heappop(pq)
        if d != dist[y, x]:
            continue
        for (dx, dy) in DIRS:
            nx, ny = x + dx, y + dy
            if not can_move(x, y, nx, ny):
                continue
            nd = d + terrain_cost(nx, ny)
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                heapq.heappush(pq, (nd, nx, ny))
    return dist

dist_blue = full_dijkstra_cost_map(start_blue)
dist_green = full_dijkstra_cost_map(start_green)

best_meeting = None
best_cost = np.inf
H, W = grid.shape
for y in range(H):
    for x in range(W):
        if not is_free(x, y):
            continue
        cb, cg = dist_blue[y, x], dist_green[y, x]
        if np.isfinite(cb) and np.isfinite(cg):
            total = cb + cg
            if total < best_cost:
                best_cost = total
                best_meeting = (x, y)

if best_meeting:
    path_blue_meet, _ = dijkstra_path(start_blue, [best_meeting])
    path_green_meet, _ = dijkstra_path(start_green, [best_meeting])
    path_meet_goal, _ = dijkstra_path(best_meeting, goal_line)
    skeleton_cost = (
        path_cost(path_blue_meet)
        + path_cost(path_green_meet)
        + path_cost(path_meet_goal)
    )
else:
    path_blue_meet, path_green_meet, path_meet_goal = [], [], []
    skeleton_cost = np.inf

# =======================
# VISUALS
# =======================
fig, ax = plt.subplots(figsize=(10,7))
ax.set_title("Random Goal Line + Skeletonization (Cost-Based with Meeting Point)")
ax.set_xlim(-0.5, GRID_W - 0.5)
ax.set_ylim(-0.5, GRID_H - 0.5)
ax.set_xticks([]); ax.set_yticks([])
ax.grid(True, color="lightgray", linestyle="-", linewidth=0.5)
ax.set_aspect("equal")
ax.invert_yaxis()

# draw terrain
for y in range(GRID_H):
    for x in range(GRID_W):
        if grid[y,x]==1:
            ax.add_patch(plt.Rectangle((x-0.5,y-0.5),1,1,color="black",zorder=6))
        else:
            col,_,a=terrain_types[terrain_map[y,x]]
            ax.add_patch(plt.Rectangle((x-0.5,y-0.5),1,1,color=to_rgba(col,a),zorder=0))

# draw red goal line
for (x,y) in goal_line:
    ax.add_patch(plt.Rectangle((x-0.5,y-0.5),1,1,color="red",alpha=0.5,zorder=5))

# draw points
for (x,y) in other_points:
    ax.plot(x,y,"o",color="gray",markersize=5,zorder=7)
ax.plot(start_blue[0],start_blue[1],"o",color="blue",markersize=10,label="Start Blue",zorder=8)
ax.plot(start_green[0],start_green[1],"o",color="green",markersize=10,label="Start Green",zorder=8)

# =======================
# STEP GENERATORS
# =======================
def dijkstra_steps(start, goals, main_color, trial_color):
    H,W = grid.shape
    dist = np.full((H,W), np.iinfo(np.int64).max, dtype=np.int64)
    parent = {}
    sx,sy=start
    pq=[(0,sx,sy)]
    dist[sy,sx]=0
    goal_set=set(goals)
    while pq:
        d,x,y=heapq.heappop(pq)
        if d!=dist[y,x]: continue
        yield ("explore",(x,y),trial_color)
        if (x,y) in goal_set:
            path=[]
            while (x,y) in parent:
                path.append((x,y))
                x,y=parent[(x,y)]
            path.append(start)
            path.reverse()
            for p in path: yield ("path",p,main_color)
            return
        for (dx,dy) in DIRS:
            nx,ny=x+dx,y+dy
            if not can_move(x,y,nx,ny): continue
            nd=d+terrain_cost(nx,ny)
            if nd<dist[ny,nx]:
                dist[ny,nx]=nd
                parent[(nx,ny)]=(x,y)
                heapq.heappush(pq,(nd,nx,ny))
                yield ("queue",(nx,ny),trial_color)

blue_steps = list(dijkstra_steps(start_blue, goal_line, "blue", "lightskyblue"))
green_steps = list(dijkstra_steps(start_green, goal_line, "green", "lightgreen"))
max_len=max(len(blue_steps),len(green_steps))
if len(blue_steps)<max_len:
    blue_steps += [("idle",(None,None),None)]*(max_len-len(blue_steps))
if len(green_steps)<max_len:
    green_steps += [("idle",(None,None),None)]*(max_len-len(green_steps))
combined_steps=[]
for b,g in zip(blue_steps,green_steps):
    combined_steps.append(("dual",(b,g)))

# =======================
# SKELETON ANIMATION (no purple dot, black Y/T)
# =======================
if best_meeting:
    # Blue to meeting
    for i in range(1, len(path_blue_meet)):
        combined_steps.append(("skeleton_line", (path_blue_meet[i-1], path_blue_meet[i]), "black"))
    # Green to meeting
    for i in range(1, len(path_green_meet)):
        combined_steps.append(("skeleton_line", (path_green_meet[i-1], path_green_meet[i]), "black"))
    # Meeting to goal
    for i in range(1, len(path_meet_goal)):
        combined_steps.append(("skeleton_line", (path_meet_goal[i-1], path_meet_goal[i]), "black"))
else:
    pass

# =======================
# DRAW
# =======================
visited_edges={}
def draw_segment(a,b,color,lw,alpha,z):
    if a and b:
        ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=lw,alpha=alpha,zorder=z)
def update(frame):
    entry=combined_steps[frame]
    if entry[0]=="dual":
        (b_type,b_pos,b_color),(g_type,g_pos,g_color)=entry[1]
        for step_type,pos,color in [(b_type,b_pos,b_color),(g_type,g_pos,g_color)]:
            if step_type=="idle" or color is None: continue
            trail=visited_edges.setdefault(color,[])
            prev=trail[-1] if trail else None
            if "explore" in step_type or "queue" in step_type:
                draw_segment(prev,pos,color,1.3,0.55,3)
            elif "path" in step_type:
                draw_segment(prev,pos,color,2.5,0.95,4)
            trail.append(pos)
    elif entry[0]=="skeleton_line":
        (a,b),color=entry[1],entry[2]
        draw_segment(a,b,color,3.0,1.0,6)
    return []

# =======================
# COSTS
# =======================
blue_cost=path_cost(path_blue)
green_cost=path_cost(path_green)

plt.subplots_adjust(right=0.8)
ax.legend(bbox_to_anchor=(1.02,0.80),loc="center left",borderaxespad=0)
ax.text(GRID_W+1,GRID_H*0.55,
        f"Blue: {blue_cost} CZK\nGreen: {green_cost} CZK\nSkeleton: {skeleton_cost:.0f} CZK",
        fontsize=11,va="center",ha="left",color="black")

anim=FuncAnimation(fig,update,frames=len(combined_steps),
                   interval=INTERVAL_MS,blit=False,repeat=False)
plt.show()
