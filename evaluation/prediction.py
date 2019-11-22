import os
import pickle


def OutPrediction(prediction, frame_number, out_directory='predictions'):
    # Save prediction as a file.
    os.makedirs(out_directory, exist_ok=True)
    out_file_name = os.path.join(out_directory, str(frame_number).zfill(4))
    with open(out_file_name, 'wb+') as out_file:
        frame = []
        for predicted_box in prediction:
            tracked_object = {}
            tracked_object['bbox'] = {'left': predicted_box['topleft']['x'], 'top': predicted_box['topleft']['y'], 'right': predicted_box['bottomright']['x'], 'bottom': predicted_box['bottomright']['y']}
            tracked_object['confidence'] = predicted_box['confidence']
            tracked_object['type'] = predicted_box['label']
            frame.append(tracked_object)

        pickle.dump(frame, out_file, pickle.HIGHEST_PROTOCOL)
