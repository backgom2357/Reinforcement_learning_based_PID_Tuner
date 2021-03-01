import cv2
import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from agent import Agent
from config import Config
from network import build_model
from utils import preprocess, normalize

def test(cf, env_name, weights_url, render=False, check_input_frames=False, check_log_plot=False, check_saliency_map=False):

    # Environment
    env_name = env_name
    env = gym.make(env_name)
    action_dim = env.action_space.n
    
    # Build Network
    model = build_model(cf.FRAME_SIZE, action_dim, cf.AGENT_HISTORY_LENGHTH)
    model.summary()
    model.load_weights(weights_url)

    # Initialize
    frames, action, done = 0, 0, 0
    initial_state = env.reset()
    state = np.stack([preprocess(initial_state, frame_size=cf.FRAME_SIZE)]*4, axis=3)
    state = np.reshape(state, state.shape[:-1])

    while not done:

        frames += 1

        # Render
        if render:
            env.render()

        # Interact with Environmnet
        action = np.argmax(model(normalize(state)))
        next_state, reward, done, _ = env.step(action)
        reward = np.clip(reward, -1, 1)
        next_state = np.append(state[..., 1:], preprocess(next_state, frame_size=cf.FRAME_SIZE), axis=3)
        state = next_state

        # Check Input
        if check_input_frames:
            test_img = np.reshape(next_state, (cf.FRAME_SIZE, cf.FRAME_SIZE, cf.AGENT_HISTORY_LENGHTH))
            test_img = cv2.resize(test_img, dsize=(300, 300), interpolation=cv2.INTER_AREA)
            cv2.imshow('input image', test_img)
            cv2.waitKey(0)!=ord('l')
            if cv2.waitKey(25)==ord('q') or done:
                cv2.destroyAllWindows()
        
        # Check Saliency Map
        if check_saliency_map:

            # Color Map
            cm = plt.get_cmap('jet')

            # Saliency Map
            grad_img = generate_grad_cam(model, state, reward, action, next_state, done, 'conv2d_6', output_layer='global_average_pooling2d')
            grad_img = np.reshape(grad_img, (cf.FRAME_SIZE, cf.FRAME_SIZE))
            grad_img = cm(grad_img)[:,:,:3]
            screen = env.render(mode='rgb_array')
            screen, grad_img = cv2.resize(screen, dsize=(400,500))/255., cv2.resize(grad_img, dsize=(400,500))
            test_img = cv2.addWeighted(screen,0.5,grad_img,0.5,0,dtype=cv2.CV_32F)
            test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
            cv2.imshow('saliency map Value', test_img)
            if cv2.waitKey(25)==ord('q') or done:
                cv2.destroyAllWindows()
        
        # Check Log & Plot
        if check_log_plot:

            # Jupter Notebook Matplotlib Setting
            is_ipython = 'inline' in matplotlib.get_backend()
            if is_ipython:
                from IPython import display
            plt.ion()

            q = np.array(model(normalize(state))[0])
            plot_durations(q, is_ipython)
            # print(action, q, end='\r')
            print(action, max(q), min(q), sum(q)/action_dim, end='\r')
            plt.ioff()


def generate_grad_cam(model, state, reward, action, next_state, done, activation_layer, output_layer):
    """
    This function generate Grad-CAM images for each frames.

    Args:
        model (tensorflow.keras.Model): trained model
        state (array-like): state
        reward (int): reward
        action (int): action
        next_state (array-like): next_state
        done (int): done
        activation_layer (str): name of activation layer
        output_layer (str): name of output layer

    Returns:
        array-like : Grad-CAM image
    """
    height, width = cf.FRAME_SIZE, cf.FRAME_SIZE
    grad_cam_model = tf.keras.models.Model(model.inputs, [model.get_layer(activation_layer).output, model.get_layer(output_layer).output])

    # Calculate gradient for weights
    with tf.GradientTape() as g:
        layer_output, pred = grad_cam_model(normalize(state))
        grad = g.gradient(pred[0][tf.argmax(pred, axis=1)[0]], layer_output)[0]
    weights = np.mean(grad, axis=(0,1))
 
    # Create Grad-CAM image
    grad_cam_image = np.zeros(dtype=np.float32, shape=layer_output.shape[1:3])
    for i, w in enumerate(weights):
        grad_cam_image += w * layer_output[0, :, :, i]

    grad_cam_image /= np.max(grad_cam_image)
    grad_cam_image = grad_cam_image.numpy()
    grad_cam_image = cv2.resize(grad_cam_image, (width, height))

    return grad_cam_image


def plot_durations(q, is_ipython):
    """
    Draw plot for the state of game

    Args:
        q (array-like): predicted q values
        is_ipython (bool): ipython or not
    """

    ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
    }

    plt.figure(2)
    plt.clf()
    plt.xlabel('Actions')
    plt.ylabel('Q value')
    action = np.argmax(q)
    mean = np.mean(q)
    normalized_q = q - mean
    plt.title(ACTION_MEANING[action])
    x = np.arange(len(q))
    xlabel = [str(a) for a in range(len(q))]
    color = ['hotpink' if i==action else 'c' for i in range(len(q))]
    plt.bar(x, normalized_q, color=color)
    plt.xticks(x, xlabel)
    plt.pause(0.01)  # 도표가 업데이트되도록 잠시 멈춤
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

if __name__ == "__main__":
    cf = Config()
    env_setting = [cf.ATARI_GAMES[6], './save_weights/bbb.h5']
    test(cf, *env_setting, render=True, check_saliency_map=True)