import json
import logging.handlers
import argparse
import os
import numpy as np

from networkx import DiGraph, all_pairs_shortest_path_length
from sklearn_hierarchical_classification.constants import ROOT
from sklearn_hierarchical_classification.metrics import multi_labeled
import sys

from config.logger import get_logger
sys.path.append('')


KEYS = ['id','labels']
logger = get_logger(__name__)

G = DiGraph()
G.add_edge(ROOT, "Logos")
G.add_edge("Logos", "Repetition")
G.add_edge("Logos", "Obfuscation, Intentional vagueness, Confusion")
G.add_edge("Logos", "Reasoning")
G.add_edge("Logos", "Justification")
G.add_edge('Justification', "Slogans")
G.add_edge('Justification', "Bandwagon")
G.add_edge('Justification', "Appeal to authority")
G.add_edge('Justification', "Flag-waving")
G.add_edge('Justification', "Appeal to fear/prejudice")
G.add_edge('Reasoning', "Simplification")
G.add_edge('Simplification', "Causal Oversimplification")
G.add_edge('Simplification', "Black-and-white Fallacy/Dictatorship")
G.add_edge('Simplification', "Thought-terminating cliché")
G.add_edge('Reasoning', "Distraction")
G.add_edge('Distraction', "Misrepresentation of Someone's Position (Straw Man)")
G.add_edge('Distraction', "Presenting Irrelevant Data (Red Herring)")
G.add_edge('Distraction', "Whataboutism")
G.add_edge(ROOT, "Ethos")
G.add_edge('Ethos', "Appeal to authority")
G.add_edge('Ethos', "Glittering generalities (Virtue)")
G.add_edge('Ethos', "Bandwagon")
G.add_edge('Ethos', "Ad Hominem")
G.add_edge('Ethos', "Transfer")
G.add_edge('Ad Hominem', "Doubt")
G.add_edge('Ad Hominem', "Name calling/Labeling")
G.add_edge('Ad Hominem', "Smears")
G.add_edge('Ad Hominem', "Reductio ad hitlerum")
G.add_edge('Ad Hominem', "Whataboutism")
G.add_edge(ROOT, "Pathos")
G.add_edge('Pathos', "Exaggeration/Minimisation")
G.add_edge('Pathos', "Loaded Language")
G.add_edge('Pathos', "Appeal to (Strong) Emotions")
G.add_edge('Pathos', "Appeal to fear/prejudice")
G.add_edge('Pathos', "Flag-waving")
G.add_edge('Pathos', "Transfer") 

def get_all_classes_from_graph(graph):
    return [
        node
        for node in graph.nodes
        if node != ROOT
        ]
    
def _h_fbeta_score(y_true, y_pred, class_hierarchy, beta=1., root=ROOT):
    hP = _h_precision_score(y_true, y_pred, class_hierarchy, root=root)
    hR = _h_recall_score(y_true, y_pred, class_hierarchy, root=root)
    return (1. + beta ** 2.) * hP * hR / (beta ** 2. * hP + hR + 0.000001)
    
def _fill_ancestors(y, graph, root, copy=True):
    y_ = y.copy() if copy else y
    paths = all_pairs_shortest_path_length(graph.reverse(copy=False))
    for target, distances in paths:
        if target == root:
            continue
        ix_rows = np.where(y[:, target] > 0)[0]
        ancestors = list(filter(lambda x: x != ROOT,distances.keys()))
        y_[tuple(np.meshgrid(ix_rows, ancestors))] = 1
    graph.reverse(copy=False)
    return y_
def _h_recall_score(y_true, y_pred, class_hierarchy, root=ROOT):
    y_true_ = _fill_ancestors(y_true, graph=class_hierarchy, root=root)
    y_pred_ = _fill_ancestors(y_pred, graph=class_hierarchy, root=root)

    # print(y_true_)
    # print(y_pred_)
    # print(class_hierarchy)

    ix = np.where((y_true_ != 0) & (y_pred_ != 0))

    true_positives = len(ix[0])
    all_positives = np.count_nonzero(y_true_)

    return true_positives / (all_positives + 0.000001)

def _h_precision_score(y_true, y_pred, class_hierarchy, root=ROOT):
    y_true_ = _fill_ancestors(y_true, graph=class_hierarchy, root=root)
    y_pred_ = _fill_ancestors(y_pred, graph=class_hierarchy, root=root)

    ix = np.where((y_true_ != 0) & (y_pred_ != 0))

    true_positives = len(ix[0])
    all_results = np.count_nonzero(y_pred_)

    return true_positives / (all_results + 0.000001)
def read_classes(file_path):
  CLASSES = []
  with open(file_path) as f:
    for label in f.readlines():
      label = label.strip()
      if label:
        CLASSES.append(label)
  return CLASSES

def check_format(file_path):
  _classes = get_all_classes_from_graph(G)
  if not os.path.exists(file_path):
    logging.error("File doesnt exists: {}".format(file_path))
    return False
  submmission = ''
  try:
    with open(file_path, encoding='utf-8') as p:
      submission = json.load(p)
  except:
    logging.error("File is not a valid json file: {}".format(file_path))
    return False
  for i, obj in enumerate(submission):
    for key in KEYS:
      if key not in obj:
        logging.error("Missing entry in {}:{}".format(file_path, i))
        return False
  for label in list(obj['labels']):
       if label not in _classes:
         # print(label)
         logging.error("Unknown Label in {}:{}".format(file_path, i))
         return False
  return True

def _read_gold_and_pred(pred_fpath, gold_fpath):
  """
  Read gold and predicted data.
  :param pred_fpath: a json file with predictions, 
  :param gold_fpath: the original annotated gold file.
  :return: {id:pred_labels} dict; {id:gold_labels} dict
  """

  gold_labels = {}
  with open(gold_fpath, encoding='utf-8') as gold_f:
    gold = json.load(gold_f)
    for obj in gold:
      gold_labels[obj['id']] = obj['labels']

  pred_labels = {}
  with open(pred_fpath, encoding='utf-8') as pred_f:
    pred = json.load(pred_f)
    for obj in pred:
      pred_labels[obj['id']] = obj['labels']

  # print(len(gold_labels), len(pred_labels))

  if set(gold_labels.keys()) != set(pred_labels.keys()):
      logger.error('There are either missing or added examples to the prediction file. Make sure you only have the gold examples in the prediction file.')
      raise ValueError('There are either missing or added examples to the prediction file. Make sure you only have the gold examples in the prediction file.')

  
  return pred_labels, gold_labels

def evaluate_h(pred_fpath, gold_fpath):

    pred_labels, gold_labels = _read_gold_and_pred(pred_fpath, gold_fpath)
  
    gold = []
    pred = []
    for id in gold_labels:
        gold.append(gold_labels[id])
        pred.append(pred_labels[id])
    with multi_labeled(gold, pred, G) as (gold_, pred_, graph_):
        # print(gold, pred, G)
        # print(gold_, pred_, graph_)
        # nx.draw(G, with_labels=True, node_color='lightblue', node_size=2000, font_size=7)
        # nx.draw(graph_, with_labels=True, node_color='lightblue', node_size=2000, font_size=7, font_weight='bold')
        # plt.show()


        return  _h_precision_score(gold_, pred_,graph_), _h_recall_score(gold_, pred_,graph_), _h_fbeta_score(gold_, pred_,graph_)
    
def validate_files(pred_files, gold_files):
  if not check_format(pred_file):
    logger.error('Bad format for pred file {}. Cannot score.'.format(pred_file))
    return False
  return True

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--gold_file_path",
    '-g',
    type=str,
    required=True,
    help="Paths to the file with gold annotations."
  )
  parser.add_argument(
    "--pred_file_path",
    '-p',
    type=str,
    required=True,
    help="Path to the file with predictions"
  )
  parser.add_argument(
    "--log_to_file",
    "-l",
    action='store_true',
    default=False,
    help="Set flag if you want to log the execution file. The log will be appended to <pred_file>.log"
  )
  args = parser.parse_args()

  pred_file = args.pred_file_path
  gold_file = args.gold_file_path

  if args.log_to_file:
    output_log_file = pred_file + ".log"
    logger.info("Logging execution to file " + output_log_file)
    fileLogger = logging.FileHandler(output_log_file)
    fileLogger.setLevel(logging.DEBUG)
    fileLogger.setFormatter(formatter)
    logger.addHandler(fileLogger)
    logger.setLevel(logging.DEBUG) 
    
  if args.log_to_file:
    logger.info('Reading gold file')
  else:
    logger.info("Reading gold predictions from file {}".format(args.gold_file_path))
  if args.log_to_file:
    logger.info('Reading predictions file')
  else:
    logger.info('Reading predictions file {}'.format(args.pred_file_path))

  if validate_files(pred_file, gold_file):
    logger.info('Prediction file format is correct')
    prec_h, rec_h, f1_h = evaluate_h(pred_file, gold_file)
    logger.info("f1_h={:.5f}\tprec_h={:.5f}\trec_h={:.5f}".format(f1_h, prec_h, rec_h))
    if args.log_to_file:
        print("{}\t{}\t{}".format(f1_h, prec_h, rec_h))
    else:
        print("f1_h={:.5f}\tprec_h={:.5f}\trec_h={:.5f}".format(f1_h, prec_h, rec_h))
    
      
