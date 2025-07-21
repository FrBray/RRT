"""
Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)
Modified to support ellipsoidal robot shapes

author: ChatGPT
"""

import math
import random
import matplotlib.pyplot as plt
import numpy as np

show_animation = True


class RRT:
    """
    Class for RRT planning supporting ellipsoidal robots
    """

    class Node:
        """
        RRT Node with orientation
        """

        def __init__(self, x, y, yaw=0.0):
            self.x = x
            self.y = y
            self.yaw = yaw
            self.path_x = []
            self.path_y = []
            self.path_yaw = []
            self.parent = None

    class AreaBounds:

        def __init__(self, area):
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=3.0,
                 path_resolution=0.5,
                 goal_sample_rate=5,
                 max_iter=500,
                 play_area=None,
                 robot_axes=(2.0, 0.5),  # semi-major (a), semi-minor (b)
                 robot_yaw=0.0):
        """
        start: [x,y]
        goal: [x,y]
        obstacle_list: [(x,y,size),...]
        rand_area: [min,max]
        robot_axes: (a,b) ellipse semi-axes
        robot_yaw: constant yaw (assumed fixed orientation)
        """
        self.start = self.Node(start[0], start[1], robot_yaw)
        self.end = self.Node(goal[0], goal[1], robot_yaw)
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.play_area = self.AreaBounds(play_area) if play_area else None
        self.robot_a, self.robot_b = robot_axes
        self.robot_yaw = robot_yaw
        self.node_list = []

    def planning(self, animation=True):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_if_outside_play_area(new_node) and \
               self.check_collision_ellipse(new_node):
                self.node_list.append(new_node)

            if animation:
                self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(new_node.x, new_node.y) <= self.expand_dis:
                final_node = self.steer(new_node, self.end, self.expand_dis)
                if self.check_collision_ellipse(final_node):
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation:
                self.draw_graph(rnd_node)
        return None

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = self.Node(from_node.x, from_node.y, from_node.yaw)
        d, theta = self.calc_distance_and_angle(from_node, to_node)
        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]
        new_node.path_yaw = [new_node.yaw]

        if extend_length > d:
            extend_length = d
        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.yaw = theta
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)
            new_node.path_yaw.append(new_node.yaw)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.path_yaw.append(theta)
            new_node.x = to_node.x
            new_node.y = to_node.y
            new_node.yaw = theta

        new_node.parent = from_node
        return new_node

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y, self.end.yaw]]
        node = self.node_list[goal_ind]
        while node.parent:
            path.append([node.x, node.y, node.yaw])
            node = node.parent
        path.append([node.x, node.y, node.yaw])
        return path

    def calc_dist_to_goal(self, x, y):
        return math.hypot(x - self.end.x, y - self.end.y)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            return self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(-math.pi, math.pi)
            )
        return self.Node(self.end.x, self.end.y, self.end.yaw)

    def draw_graph(self, rnd=None):
        # plt.clf()
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: exit(0) if event.key == 'escape' else None)
        if rnd:
            plt.plot(rnd.x, rnd.y, "^k")
            self.plot_ellipse(rnd.x, rnd.y, self.robot_a, self.robot_b, rnd.yaw, '-r')
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")
        for (ox, oy, (len, wid)) in self.obstacle_list:
            self.plot_rectangle(ox, oy, len, wid)
        if self.play_area:
            plt.plot(
                [self.play_area.xmin, self.play_area.xmax, self.play_area.xmax, self.play_area.xmin, self.play_area.xmin],
                [self.play_area.ymin, self.play_area.ymin, self.play_area.ymax, self.play_area.ymax, self.play_area.ymin],
                "-k"
            )
        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
        plt.grid(True)
        plt.pause(0.1)

    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5)) + [0]
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    @staticmethod
    def plot_rectangle(x, y, length, width, color="-b"):  # pragma: no cover
        dx = length / 2.0
        dy = width  / 2.0
        xs = [x - dx, x + dx, x + dx, x - dx, x - dx]
        ys = [y - dy, y - dy, y + dy, y + dy, y - dy]
        plt.plot(xs, ys, color)

    @staticmethod
    def plot_ellipse(x, y, a, b, yaw=0.0, color="-b"):  # pragma: no cover
        degs = np.linspace(0, 360, num=72)
        pts = np.array([[a * math.cos(math.radians(d)), b * math.sin(math.radians(d))] for d in degs])
        R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        pts = pts.dot(R.T)
        xs = pts[:, 0] + x
        ys = pts[:, 1] + y
        plt.plot(xs, ys, color)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 for node in node_list]
        return dlist.index(min(dlist))

    def check_if_outside_play_area(self, node):
        if not self.play_area:
            return True
        return (self.play_area.xmin <= node.x <= self.play_area.xmax and
                self.play_area.ymin <= node.y <= self.play_area.ymax)

    def check_collision_ellipse(self, node):
        """
        Check collision of an ellipse centered at each path point
        """
        a, b, yaw = self.robot_a, self.robot_b, self.robot_yaw
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        for (px, py) in zip(node.path_x, node.path_y):
            # bounding box
            dx = abs(a * cos_y) + abs(b * sin_y)
            dy = abs(a * sin_y) + abs(b * cos_y)
            # check each obstacle
            for (ox, oy, (len, wid)) in self.obstacle_list:
                if abs(px - ox) > dx + len or abs(py - oy) > dy + wid:
                    continue
                # transform point
                dx_w = px - ox
                dy_w = py - oy
                dx_l = dx_w * cos_y + dy_w * sin_y
                dy_l = -dx_w * sin_y + dy_w * cos_y
                if (dx_l / (a + len))**2 + (dy_l / (b + wid))**2 <= 1.0:
                    return False
        return True

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta


def main(gx=6.0, gy=10.0):
    print("start " + __file__)
    obstacleList = [(5, 5, (1,3)) , (3, 6, (1,2)), (3, 8, (2,2)), (3, 10, (1,2)), (7, 5, (3,1)),]
                    #(9, 5, 2), (8, 10, 1)]  # [x, y, radius]
    rrt = RRT(
        start=[0, 0],
        goal=[gx, gy],
        rand_area=[-2, 15],
        obstacle_list=obstacleList,
        robot_axes=(0.8, 0.4),  # ellipse semi-axes
        play_area=None
    )
    path = rrt.planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")
        if show_animation:
            rrt.draw_graph()
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            plt.plot(xs, ys, '-r')
            plt.grid(True)
            plt.pause(0.1)
            plt.show()

if __name__ == '__main__':
    main()
