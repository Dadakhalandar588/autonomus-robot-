import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid


class MapReader(Node):

    def __init__(self):
        super().__init__('map_reader')

        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.map_received = False

    def map_callback(self, msg):

        if self.map_received:
            return

        self.map_received = True

        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution

        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        data = msg.data

        free = data.count(0)
        obstacles = data.count(100)
        unknown = data.count(-1)

        print("\n========== MAP RECEIVED ==========")
        print("Width      :", width)
        print("Height     :", height)
        print("Resolution :", resolution)
        print("Origin X   :", origin_x)
        print("Origin Y   :", origin_y)

        print("\n========== CELL COUNT ==========")
        print("Free       :", free)
        print("Obstacles  :", obstacles)
        print("Unknown    :", unknown)
        print("Total      :", len(data))

        # Convert 1D data into 2D grid
        grid = []

        for y in range(height):
            row = []

            for x in range(width):
                index = y * width + x
                row.append(data[index])

            grid.append(row)

        print("\n========== FIRST 5 ROWS ==========")

        for row in grid[:5]:
            print(row)


def main(args=None):

    rclpy.init(args=args)

    node = MapReader()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()