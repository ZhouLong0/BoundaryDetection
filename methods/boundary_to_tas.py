def create_step_annotation_list(matched_data):
    """
    Transforms a timeline dictionary into a dense frame-by-frame annotation list.
    Validates that every frame is explicitly covered by the timeline.
    
    Args:
        matched_data (dict): Dict containing 'num_frames' and 'timeline'.
        
    Returns:
        list: A list of strings with length equal to num_frames.
    """
    num_frames = matched_data.get('num_frames', 0)
    timeline = matched_data.get('timeline', [])
    
    # Initialize the list with a distinct placeholder
    PLACEHOLDER = "UNFILLED"
    annotations = [PLACEHOLDER] * num_frames
    
    for i, event in enumerate(timeline):
        # Determine boundaries
        start_frame = event.get('start_frame', 0)
    
        if i + 1 < len(timeline):
            end_frame = timeline[i+1].get('start_frame', num_frames)
        else:
            end_frame = num_frames
            
        # Clamp to avoid index out-of-bounds
        start_frame = max(0, min(start_frame, num_frames))
        end_frame = max(0, min(end_frame, num_frames))
        
        # Grab the text directly from the matched data
        step_text = event.get('matched_step', 'BG')
        
        # Fill the frames for this event
        for f in range(start_frame, end_frame):
            annotations[f] = step_text
            
    # Check if any placeholders remain
    if PLACEHOLDER in annotations:
        unfilled_count = annotations.count(PLACEHOLDER)
        unfilled_indices = [idx for idx, val in enumerate(annotations) if val == PLACEHOLDER]
        
        # You can choose to raise an error or just print a warning:
        raise ValueError(
            f"Validation Failed: {unfilled_count} frames were not covered by the timeline. "
            f"Unfilled indices start at: {unfilled_indices[:10]}..."
        )
        
    return annotations



def _evaluate_tas_predictions(dataset, result_dir, exp_id=0, dataset_name="egoper"):
    import numpy as np
    from other_methods.ProTAS.eval import read_file, edit_score, f_score
    import json
    import os

    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    correct = 0
    total = 0
    correct_wo_bg = 0
    total_wo_bg = 0
    edit = 0

    dataset_name = str(dataset_name).lower()
    if dataset_name == 'egoper':
        bg_class = ['BG']
    elif dataset_name == 'gtea':
        bg_class = ['BG']
    elif dataset_name == 'breakfast':
        bg_class = ['BG', 'SIL']
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    for video in dataset:
        video_id = video['video_id']
        recipe = video.get('recipe', '')

        if dataset_name == 'egoper':
            ground_truth_path = os.path.join("data/Egoper/frame_annotations", recipe)
            recog_path = os.path.join(result_dir, recipe)
        elif dataset_name == 'gtea':
            ground_truth_path = "data/GTEA/data/gtea/groundTruth"
            recog_path = result_dir
        else:  # breakfast
            ground_truth_path = "data/Breakfast/groundTruth"
            recog_path = result_dir

        gt_file = os.path.join(ground_truth_path, video_id + ".txt")
        gt_content = read_file(gt_file).split('\n')[0:-1]
        
    
        with open(os.path.join(recog_path, video_id + "_matched_tas.json"), 'r') as f:
            recog_content = json.load(f)

        if len(gt_content) != len(recog_content):
            print(
                f"[Length Mismatch] {video_id}: "
                f"GT={len(gt_content)} vs Pred={len(recog_content)}. "
                f"Evaluating on min length={min(len(gt_content), len(recog_content))}."
            )

        min_len = min(len(gt_content), len(recog_content))
        gt_content = gt_content[:min_len]
        recog_content = recog_content[:min_len]

        print(f"Evaluating {video_id} with {len(gt_content)} GT steps and {len(recog_content)} recognized steps.")

        for i in range(len(gt_content)):
            if gt_content[i] not in bg_class:
                total_wo_bg += 1
                if gt_content[i] == recog_content[i]:
                    correct_wo_bg += 1
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1
        
        edit += edit_score(recog_content, gt_content, bg_class=bg_class)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s], bg_class)
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
            
    if total == 0:
        print(f"No samples available for {dataset_name} evaluation.")
        return

    acc = 100*float(correct)/total
    acc_wo_bg = 100*float(correct_wo_bg)/max(total_wo_bg, 1)
    edit = (1.0*edit)/max(len(dataset), 1)
    res_list = [acc, acc_wo_bg, edit]

    #print("Acc: %.4f" % (100*float(correct)/total))
    #print('Edit: %.4f' % ((1.0*edit)/len(list_of_videos)))
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s]+fp[s]) if (tp[s] + fp[s]) > 0 else 0.0
        recall = tp[s] / float(tp[s]+fn[s]) if (tp[s] + fn[s]) > 0 else 0.0

        f1 = 2.0 * (precision*recall) / (precision+recall) if (precision + recall) > 0 else 0.0

        f1 = np.nan_to_num(f1)*100
        #print('F1@%0.2f: %.4f' % (overlap[s], f1))
        res_list.append(f1)
    print(exp_id, ' '.join(['{:.2f}'.format(r) for r in res_list]))
    result_metrics = {'exp_id': exp_id, 'Acc': acc,  'Acc-bg': acc_wo_bg, 'Edit': edit, 
                    'F1@10': res_list[-3], 'F1@25': res_list[-2], 'F1@50': res_list[-1]}
    print(f"Evaluation Metrics for Experiment {exp_id}: {result_metrics}")

    result_path = os.path.join(recog_path, 'eval.json')
    with open(result_path, 'w') as fw:
        json.dump(result_metrics, fw, indent=4)
    print(f"Saved evaluation results to {result_path}")


def evaluate_egoper(dataset, result_dir, exp_id=0):
    return _evaluate_tas_predictions(dataset=dataset, result_dir=result_dir, exp_id=exp_id, dataset_name='egoper')


def evaluate_gtea(dataset, result_dir, exp_id=0):
    return _evaluate_tas_predictions(dataset=dataset, result_dir=result_dir, exp_id=exp_id, dataset_name='gtea')


def evaluate_breakfast(dataset, result_dir, exp_id=0):
    return _evaluate_tas_predictions(dataset=dataset, result_dir=result_dir, exp_id=exp_id, dataset_name='breakfast')


def evaluate_tas(dataset, result_dir, dataset_name, exp_id=0):
    return _evaluate_tas_predictions(dataset=dataset, result_dir=result_dir, exp_id=exp_id, dataset_name=dataset_name)