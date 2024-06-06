class NascarTrackDriveWaypoints:
    def __init__(self):
        self.tracks = self._initialize_tracks()

    def get_track(self, track_index):
        if track_index < 0 or track_index >= len(self.tracks):
            print("###### Track index out of range - setting to 0! ########")
            track_index = 0

        track = self.tracks[track_index]
        absolute_start_position = track["absolute_start_position"]
        stop_position = track["stop_position"]
        waypoint_and_rotation = track["waypoint_and_rotation"]

        waypoints = [x[0] for x in waypoint_and_rotation]
        # Subtract the absolute start position to get relative waypoints
        waypoints = [(w[0], w[1]) for w
                     in waypoints]
        rotations = [x[1] for x in waypoint_and_rotation]

        # create relative stop position
        stop_position = (
        (stop_position[0][0] - absolute_start_position[0][0], stop_position[0][1] - absolute_start_position[0][1]),
        stop_position[1])

        return waypoints, rotations, stop_position

    def _initialize_tracks(self):
        return [
            { # LEFT
                "absolute_start_position": ((172.8, 276.8), 355.0),
                "stop_position": ((0, 0), 0),
                "waypoint_and_rotation": [
                    ((-0.6, -14.0), 358.6),
                    ((-3.6, -21.6), 315.5),
                    ((-10.8, -25.4), 270.2),
                    ((-44.4, -25.2), 267.0),
                    ((-55.3, -20.0), 224.0),
                    ((-60.2, -9.4), 184.5),
                    ((-59.2, 2.7), 152.6),
                    ((-48.8, 12.2), 104.7),
                    ((-37.3, 14.2), 91.8),
                    ((-17.0, 14.5), 88.7),
                    ((-8.6, 12.5), 55.4),
                    ((-1.3, 2.3), 15.7),
                ]
            },
            { # RIGHT
                "absolute_start_position": ((180.4, 238.7), 270.0),
                "stop_position": ((0,0), 0),
                "waypoint_and_rotation": [
                    ((-20.2, 0.5), 265.3),
                    ((-33.5, 0.9), 275.6),
                    ((-45.1, -7.6), 338.5),
                    ((-47.1, -18.7), 356.6),
                    ((-47.3, -30.0), 0.7),
                    ((-47.1, -42.0), 1.5),
                    ((-40.6, -49.0), 91.6),
                    ((-33.1, -48.5), 87.8),
                    ((-22.1, -49.0), 91.2),
                    ((-8.7, -48.7), 88.1),
                    ((6.2, -46.1), 127.7),
                    ((10.7, -36.9), 172.6),
                    ((11.1, -19.5), 178.1),
                    ((9.6, -5.6), 214.7),
                    ((-0.4, -0.3), 268.1),
                ]
            },
            {  # LEFT
                "absolute_start_position": ((191.7, 401.3), 355.0),
                "stop_position": ((0, 0), 0),
                "waypoint_and_rotation": [
                    ((-0.6, -14.0), 358.6),
                    ((-3.6, -21.6), 315.5),
                    ((-10.8, -25.4), 270.2),
                    ((-44.4, -25.2), 267.0),
                    ((-55.3, -20.0), 224.0),
                    ((-60.2, -9.4), 184.5),
                    ((-59.2, 2.7), 152.6),
                    ((-48.8, 12.2), 104.7),
                    ((-37.3, 14.2), 91.8),
                    ((-17.0, 14.5), 88.7),
                    ((-8.6, 12.5), 55.4),
                    ((-1.3, 2.3), 15.7),
                ]
            },
            {  # LEFT
                "absolute_start_position": ((191.7, 449.7), 355.0),
                "stop_position": ((0, 0), 0),
                "waypoint_and_rotation": [
                    ((-0.6, -14.0), 358.6),
                    ((-3.6, -21.6), 315.5),
                    ((-10.8, -25.4), 270.2),
                    ((-44.4, -25.2), 267.0),
                    ((-55.3, -20.0), 224.0),
                    ((-60.2, -9.4), 184.5),
                    ((-59.2, 2.7), 152.6),
                    ((-48.8, 12.2), 104.7),
                    ((-37.3, 14.2), 91.8),
                    ((-17.0, 14.5), 88.7),
                    ((-8.6, 12.5), 55.4),
                    ((-1.3, 2.3), 15.7),
                ]
            },
            {  # LEFT
                "absolute_start_position": ((119.6, 485.1), 355.0),
                "stop_position": ((0, 0), 0),
                "waypoint_and_rotation": [
                    ((-0.6, -14.0), 358.6),
                    ((-3.6, -21.6), 315.5),
                    ((-10.8, -25.4), 270.2),
                    ((-44.4, -25.2), 267.0),
                    ((-55.3, -20.0), 224.0),
                    ((-60.2, -9.4), 184.5),
                    ((-59.2, 2.7), 152.6),
                    ((-48.8, 12.2), 104.7),
                    ((-37.3, 14.2), 91.8),
                    ((-17.0, 14.5), 88.7),
                    ((-8.6, 12.5), 55.4),
                    ((-1.3, 2.3), 15.7),
                ]
            },
            {  # LEFT
                "absolute_start_position": ((119.6, 433.0), 355.0),
                "stop_position": ((0, 0), 0),
                "waypoint_and_rotation": [
                    ((-0.6, -14.0), 358.6),
                    ((-3.6, -21.6), 315.5),
                    ((-10.8, -25.4), 270.2),
                    ((-44.4, -25.2), 267.0),
                    ((-55.3, -20.0), 224.0),
                    ((-60.2, -9.4), 184.5),
                    ((-59.2, 2.7), 152.6),
                    ((-48.8, 12.2), 104.7),
                    ((-37.3, 14.2), 91.8),
                    ((-17.0, 14.5), 88.7),
                    ((-8.6, 12.5), 55.4),
                    ((-1.3, 2.3), 15.7),
                ]
            },
            {  # RIGHT
                "absolute_start_position": ((33.6, 499.0), 270.0),
                "stop_position": ((0, 0), 0),
                "waypoint_and_rotation": [
                    ((-20.2, 0.5), 265.3),
                    ((-33.5, 0.9), 275.6),
                    ((-45.1, -7.6), 338.5),
                    ((-47.1, -18.7), 356.6),
                    ((-47.3, -30.0), 0.7),
                    ((-47.1, -42.0), 1.5),
                    ((-40.6, -49.0), 91.6),
                    ((-33.1, -48.5), 87.8),
                    ((-22.1, -49.0), 91.2),
                    ((-8.7, -48.7), 88.1),
                    ((6.2, -46.1), 127.7),
                    ((10.7, -36.9), 172.6),
                    ((11.1, -19.5), 178.1),
                    ((9.6, -5.6), 214.7),
                    ((-0.4, -0.3), 268.1),
                ]
            },
            {  # RIGHT
                "absolute_start_position": ((33.6, 440.2), 270.0),
                "stop_position": ((0, 0), 0),
                "waypoint_and_rotation": [
                    ((-20.2, 0.5), 265.3),
                    ((-33.5, 0.9), 275.6),
                    ((-45.1, -7.6), 338.5),
                    ((-47.1, -18.7), 356.6),
                    ((-47.3, -30.0), 0.7),
                    ((-47.1, -42.0), 1.5),
                    ((-40.6, -49.0), 91.6),
                    ((-33.1, -48.5), 87.8),
                    ((-22.1, -49.0), 91.2),
                    ((-8.7, -48.7), 88.1),
                    ((6.2, -46.1), 127.7),
                    ((10.7, -36.9), 172.6),
                    ((11.1, -19.5), 178.1),
                    ((9.6, -5.6), 214.7),
                    ((-0.4, -0.3), 268.1),
                ]
            },
            {  # RIGHT
                "absolute_start_position": ((-55.9, 568.4), 270.0),
                "stop_position": ((0, 0), 0),
                "waypoint_and_rotation": [
                    ((-20.2, 0.5), 265.3),
                    ((-33.5, 0.9), 275.6),
                    ((-45.1, -7.6), 338.5),
                    ((-47.1, -18.7), 356.6),
                    ((-47.3, -30.0), 0.7),
                    ((-47.1, -42.0), 1.5),
                    ((-40.6, -49.0), 91.6),
                    ((-33.1, -48.5), 87.8),
                    ((-22.1, -49.0), 91.2),
                    ((-8.7, -48.7), 88.1),
                    ((6.2, -46.1), 127.7),
                    ((10.7, -36.9), 172.6),
                    ((11.1, -19.5), 178.1),
                    ((9.6, -5.6), 214.7),
                    ((-0.4, -0.3), 268.1),
                ]
            },
            {  # RIGHT
                "absolute_start_position": ((-55.9, 503.1), 270.0),
                "stop_position": ((0, 0), 0),
                "waypoint_and_rotation": [
                    ((-20.2, 0.5), 265.3),
                    ((-33.5, 0.9), 275.6),
                    ((-45.1, -7.6), 338.5),
                    ((-47.1, -18.7), 356.6),
                    ((-47.3, -30.0), 0.7),
                    ((-47.1, -42.0), 1.5),
                    ((-40.6, -49.0), 91.6),
                    ((-33.1, -48.5), 87.8),
                    ((-22.1, -49.0), 91.2),
                    ((-8.7, -48.7), 88.1),
                    ((6.2, -46.1), 127.7),
                    ((10.7, -36.9), 172.6),
                    ((11.1, -19.5), 178.1),
                    ((9.6, -5.6), 214.7),
                    ((-0.4, -0.3), 268.1),
                ]
            },
        ]