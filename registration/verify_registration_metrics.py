"""Synthetic checks for true LOO predictions and persisted stage reports."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from registration_metrics import calculate_registration_metrics, predictive_loo_rmse, RMSELOO_METHOD
from sar_registration.geometry import project, weighted_affine
from sar_registration.learned_matching import predictive_loo
from sar_registration.reporting import export_final, IterationRecorder
from sar_registration.sift_fusion import FusionResult, RepresentationMatch
from sar_registration.baseline_validation import input_fingerprint, load_comparable_baseline


def press_rmse(source, target, weights):
    """Independent linear-model PRESS identity, without leave-one-out refits."""
    design = np.column_stack((source, np.ones(len(source))))
    inverse = np.linalg.inv(design.T @ (weights[:, None] * design))
    coefficients = inverse @ design.T @ (weights[:, None] * target)
    leverage = weights * np.einsum('ij,jk,ik->i', design, inverse, design)
    prediction_errors = (design @ coefficients - target) / (1 - leverage[:, None])
    return float(np.sqrt(np.mean(np.sum(prediction_errors ** 2, axis=1))))


class RegistrationMetricsTests(unittest.TestCase):
    def setUp(self):
        self.source = np.array([[80.,80.],[120.,80.],[160.,80.],[80.,120.],
                                [160.,120.],[80.,160.],[120.,160.],[160.,160.]])
        transform = np.array([[1.03,.02,2.],[-.01,.98,1.],[0.,0.,1.]])
        self.exact = project(self.source, transform)
        self.target = self.exact + np.array([[.2,-.1],[-.3,.2],[.8,.3],[.1,-.2],
                                             [-.2,.1],[.5,-.4],[-.1,.2],[1.3,-.7]])
        self.weights = np.array([.2,1.,.4,.7,1.,.3,.8,.5])
        self.rgb = np.zeros((256,256,3), np.uint8)

    def metrics(self, source, target, weights=None):
        w = np.ones(len(source)) if weights is None else weights
        transform = weighted_affine(source, target, w)
        errors = np.linalg.norm(project(source, transform) - target, axis=1)
        return calculate_registration_metrics(target, errors, (256,256),
                                              source_points=source, weights=weights), errors

    def test_exact_affine_prediction(self):
        metrics, _ = self.metrics(self.source, self.exact)
        self.assertLess(metrics['RMSEloo'], 1e-9)
        self.assertEqual(metrics['RMSEloo_method'], RMSELOO_METHOD)

    def test_equal_and_confidence_weights_match_press(self):
        for weights in (np.ones(8), self.weights):
            with self.subTest(weights=weights.tolist()):
                metrics, errors = self.metrics(self.source, self.target, weights)
                self.assertAlmostEqual(metrics['RMSEloo'], press_rmse(self.source, self.target, weights), places=10)
                self.assertAlmostEqual(metrics['RMSEloo'], predictive_loo(self.source, self.target, weights), places=12)
                legacy = np.mean(np.sqrt((np.sum(errors**2) - errors**2) / 7))
                self.assertGreater(metrics['RMSEloo'], legacy)
                self.assertAlmostEqual(metrics['RMSEall'], np.sqrt(np.mean(errors**2)))

    def test_empty_insufficient_and_degenerate_sets(self):
        for source in (np.empty((0,2)), self.source[:3],
                       np.array([[0.,0.],[1.,0.],[2.,0.],[0.,1.]])):
            target = source + [2.,1.]
            metrics = calculate_registration_metrics(target, np.zeros(len(source)), (256,256), source_points=source)
            self.assertIsNone(metrics['RMSEloo'])
            json.dumps(metrics, allow_nan=False)
            self.assertTrue(np.isinf(predictive_loo_rmse(source, target)))

    def test_invalid_point_and_weight_counts(self):
        with self.assertRaises(ValueError):
            calculate_registration_metrics(self.target, np.zeros(8), (256,256), source_points=self.source[:7])
        with self.assertRaises(ValueError):
            predictive_loo_rmse(self.source, self.target, np.ones(7))

    def test_final_and_iteration_reports_preserve_retained_weights(self):
        mask = np.arange(8) != 2
        expected = press_rmse(self.source[mask], self.target[mask], self.weights[mask])
        transform = weighted_affine(self.source[mask], self.target[mask], self.weights[mask])
        with tempfile.TemporaryDirectory(prefix='sar_loo_reports_') as temporary:
            root = Path(temporary)
            metrics = export_final(root/'final', self.rgb, self.rgb, transform,
                                   self.source, self.target, mask, 'test', scores=self.weights)
            self.assertAlmostEqual(metrics['RMSEloo'], expected, places=10)
            saved = json.loads((root/'final/metrics.json').read_text())
            self.assertEqual(saved['RMSEloo'], metrics['RMSEloo'])
            csv = np.loadtxt(root/'final/matched_points.csv', delimiter=',', skiprows=1)
            np.testing.assert_array_equal(csv[:,6], self.weights)
            retained = csv[:,4] > 0
            recomputed = calculate_registration_metrics(csv[retained,2:4], csv[retained,5], (256,256),
                                                        source_points=csv[retained,:2], weights=csv[retained,6])
            self.assertAlmostEqual(recomputed['RMSEloo'], saved['RMSEloo'], places=12)
            record = dict(iteration=1, status='accepted', accepted=True,
                          candidate_transform=transform.tolist(), evaluated_transform='candidate',
                          input_transform=np.eye(3).tolist(), output_transform=transform.tolist())
            event = dict(record=record, source=self.source, target=self.target, mask=mask,
                         scores=self.weights, evaluation_transform=transform, input_transform=np.eye(3),
                         output_transform=transform, source_ids=np.arange(8), original_points=self.source,
                         projected=project(self.source, transform))
            artifact = IterationRecorder(root/'iterations', self.rgb, self.rgb)(event)
            self.assertAlmostEqual(artifact['metrics']['RMSEloo'], expected, places=10)
            text = (root/'final/registration_metrics.txt').read_text()
            self.assertIn('RMSEloo='+str(metrics['RMSEloo']), text)

    def test_sift_stage_and_comparison_reports(self):
        from sar_registration.reporting import comparison
        transform = weighted_affine(self.source, self.target, np.ones(8))
        with tempfile.TemporaryDirectory(prefix='sar_loo_sift_') as temporary:
            root = Path(temporary)
            image_path = root/'image.png'
            Image.fromarray(self.rgb).save(image_path)
            match = RepresentationMatch(0, 'Original', str(image_path), transform,
                                        self.source, self.target, np.ones(8,bool), 1., .5, .1, 1.)
            result = FusionResult(transform, self.source, self.target, self.weights, np.ones(8,int),
                                  ['original']*8, np.ones(8), [1.], 0, [match], [0], [image_path], ['Original'])
            result.save(root/'fusion', reference_path=image_path)
            report = json.loads((root/'fusion/fusion_report.json').read_text())
            expected = press_rmse(self.source, self.target, np.ones(8))
            self.assertAlmostEqual(report['sift_metrics']['Original']['RMSEloo'], expected, places=10)
            final = export_final(root/'final', self.rgb, self.rgb, transform, self.source,
                                 self.target, np.ones(8,bool), 'test', scores=self.weights)
            rows = comparison(root/'fusion', root/'final', final)
            self.assertAlmostEqual(rows[0]['RMSEloo'], expected, places=10)
            self.assertAlmostEqual(rows[1]['RMSEloo'], press_rmse(self.source, self.target, self.weights), places=10)

    def test_historical_baseline_requires_matching_metric_definition(self):
        with tempfile.TemporaryDirectory(prefix='sar_loo_baseline_') as temporary:
            root = Path(temporary)
            inputs = [root/'reference', root/'sensed']
            for path in inputs:
                path.write_bytes(b'synthetic')
            directory = root/'baseline_reference'
            directory.mkdir()
            (directory/'input_sha256.json').write_text(json.dumps(input_fingerprint(inputs)))
            (directory/'metrics.json').write_text(json.dumps({'RMSEloo': .1}))
            self.assertIsNone(load_comparable_baseline(root, inputs))
            (directory/'metrics.json').write_text(json.dumps({'RMSEloo': .1, 'RMSEloo_method': RMSELOO_METHOD}))
            self.assertEqual(load_comparable_baseline(root, inputs)['RMSEloo'], .1)


if __name__ == '__main__':
    unittest.main()
