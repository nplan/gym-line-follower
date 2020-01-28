from setuptools import setup

setup(name='gym_line_follower',
      version='0.1.1',
      install_requires=['gym==0.10.8',
                        'pybullet', 'opencv-python', 'shapely', 'numpy'],
      author="Nejc Planinsek",
      author_email="planinseknejc@gmail.com",
      description="Line follower simulator environment.",
      )
