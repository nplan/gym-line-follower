from setuptools import setup

setup(name='gym_line_follower',
      version='0.1.2',
      install_requires=['gym',
                        'pybullet', 'opencv-python', 'shapely', 'numpy'],
      author="Nejc Planinsek",
      author_email="planinseknejc@gmail.com",
      description="Line follower simulator environment.",
      )
