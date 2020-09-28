import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math

try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

from yolo import YOLO
from PIL import Image

yolo = YOLO()

IM_WIDTH = 480
IM_HEIGHT = 360
Show_path_trajectory = False

red = carla.Color(255, 0, 0)
green = carla.Color(0, 255, 0)
blue = carla.Color(47, 210, 231)
cyan = carla.Color(0, 255, 255)
yellow = carla.Color(255, 255, 0)
orange = carla.Color(255, 162, 0)
white = carla.Color(255, 255, 255)

def draw_transform(debug, trans, col=carla.Color(255, 0, 0), lt=-1):
    debug.draw_arrow(
    trans.location, trans.location + trans.get_forward_vector(),
    thickness=0.05, arrow_size=0.1, color=col, life_time=lt)

def draw_waypoint_union(debug, w0, w1, color=carla.Color(255, 0, 0), lt=0.5):
    debug.draw_line(
    w0 + carla.Location(z=0.25),
    w1 + carla.Location(z=0.25),
    thickness=0.1, color=color, life_time=lt, persistent_lines=False)
    debug.draw_point(w1 + carla.Location(z=0.25), 0.105, color, lt, False)

def process_img(image, c):
    if c == 'seg':
        image.convert(carla.ColorConverter.CityScapesPalette)

    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]

    if c == 'rgb':
        # image = Image.fromarray(cv2.cvtColor(i3,cv2.COLOR_BGR2RGB))
        image = Image.fromarray(np.uint8(cv2.cvtColor(i3,cv2.COLOR_BGR2RGB)))
        r_image = yolo.detect_image(image)
        i4 = cv2.cvtColor(np.asarray(r_image),cv2.COLOR_RGB2BGR)
        cv2.imshow('obj', i4)

    cv2.imshow(c, i3)
    
    cv2.waitKey(1)
    return i3/255.0

def process_img_rgb(image, c):

    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]

    image = Image.fromarray(cv2.cvtColor(i3,cv2.COLOR_BGR2RGB))  
    r_image = yolo.detect_image(image)
    i4 = cv2.cvtColor(np.asarray(r_image),cv2.COLOR_RGB2BGR)  

    cv2.imshow(c, i3)
    cv2.imshow('obj', i4)
    cv2.waitKey(1)
    return i3/255.0

def process_img_seg(image, c):

    image.convert(carla.ColorConverter.CityScapesPalette)

    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]

    cv2.imshow(c, i3)
    cv2.waitKey(1)
    return i3/255.0

actor_list = []

try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)
    # world = client.get_world()
    world = client.load_world('Town05')
    debug = world.debug

    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter("model3")[0]
    print(bp)

    spawn_point = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.set_autopilot(True)
    # vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    actor_list.append(vehicle)

    #camera_bp = blueprint_library.find('sensor.camera.rgb')
    #camera = world.spawn_actor(camera_bp, relative_transform, attach_to=my_vehicle)
    #camera.listen(lambda image: image.save_to_disk('output/%06d.png' % image.frame_number))

    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    cam_bp.set_attribute("fov", "110")

    #spawn_point = carla.Transform(carla.Location(x=2.5, z=0.5))
    spawn_point = carla.Transform(carla.Location(x=2.5, z=2.5), carla.Rotation(pitch=-30))

    sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
    actor_list.append(sensor)
    sensor.listen(lambda data: process_img(data, 'rgb'))

    camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    camera_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    camera_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    camera_bp.set_attribute("fov", "110")
    camera = world.spawn_actor(camera_bp, spawn_point, attach_to=vehicle)
    actor_list.append(camera)
    camera.listen(lambda data: process_img(data, 'seg'))

    time.sleep(120)

    if Show_path_trajectory:
        current_ = vehicle.get_location()
        while True:
            next_ = vehicle.get_location()
            # vector = vehicle.get_velocity()

            draw_waypoint_union(debug, current_, next_, green, 30)
            debug.draw_string(current_, str('%15.0f' % (math.sqrt((next_.x - current_.x)**2 + (next_.y - current_.y)**2 + (next_.z - current_.z)**2))), False, orange, 30)

            current_ = next_
            time.sleep(1)

finally:
    for actor in actor_list:
        actor.destroy()
        #carla.command.DestroyActor(actor)
    print("All cleaned up!")