/**
 * The MIT License (MIT)
 *
 * Copyright (c) 2025 Toha <tohenk@yahoo.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const faceLandmarksDetection = require('@tensorflow-models/face-landmarks-detection');
const LocalModel = require('./model');

/**
 * Provides face landmarks detection.
 *
 * @author Toha <tohenk@yahoo.com>
 */
class FaceDetection {

    /**
     * Constructor.
     *
     * @param {object} options Options
     */
    constructor(options) {
        options = options || {};
        this.refineLandmarks = options.refineLandmarks !== undefined ?
            options.refineLandmarks : true;
        this.tfjsBackend = options.tfjsBackend ?? 'cpu';
    }

    /**
     * Get face detector configuration.
     *
     * @returns {Promise<object>}
     */
    async getConfig() {
        const faceDetectionModel = await LocalModel.create('face-detection', 'short');
        const faceLandmarkModel = await LocalModel.create('face-landmarks-detection',
            this.refineLandmarks ? 'attention-mesh' : 'face-mesh');
        return {
            runtime: 'tfjs',
            refineLandmarks: this.refineLandmarks,
            detectorModelUrl: faceDetectionModel,
            landmarkModelUrl: faceLandmarkModel,
        }
    }

    /**
     * Create face landmarks detector.
     *
     * @returns {Promise<faceLandmarksDetection.FaceLandmarksDetector>}
     */
    async getDetector() {
        const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
        return await faceLandmarksDetection.createDetector(model, await this.getConfig());
    }

    /**
     * Detect face landmarks.
     *
     * @param {string|Buffer} input Face image data
     * @returns {Promise<Array>}
     */
    async getFace(input) {
        let res, backend;
        if (this.detector === undefined) {
            this.detector = await this.getDetector();
        }
        if (typeof input === 'string' && fs.existsSync(input)) {
            input = fs.readFileSync(input);
        }
        const img = tf.node.decodeImage(input, 3); // needs RGB;
        try {
            if (img) {
                if (this.tfjsBackend && this.tfjsBackend !== tf.getBackend()) {
                    backend = tf.getBackend();
                    await tf.setBackend(this.tfjsBackend);
                }
                this.detector.reset();
                const faces = await this.detector.estimateFaces(img, {flipHorizontal: false});
                if (faces.length) {
                    res = faces[0];
                    res.shape = img.shape;
                }
            }
        }
        catch (err) {
            console.error(err);
        }
        if (img) {
            tf.dispose(img);
        }
        if (backend) {
            await tf.setBackend(backend);
        }
        return res;
    }
}

/**
 * Represents a face landmarks.
 *
 * @author Toha <tohenk@yahoo.com>
 */
class FaceLandmark {

    markers = {
        faceOval: false,
        leftEye: true,
        leftEyebrow: false,
        leftIris: true,
        lips: true,
        rightEye: true,
        rightEyebrow: false,
        rightIris: true,
    }
    scale = 1

    constructor({shape, box, keypoints}) {
        if (shape !== undefined && shape !== null) {
            this.shape = shape;
        }
        this.box = box;
        this.points = Points.from(keypoints);
        this.points.normalize({
            xMin: this.box.xMin,
            yMin: this.box.yMin,
            zMin: this.points.getMin('z'),
            xScale: this.scale / this.box.width,
            yScale: this.scale / this.box.height,
            zScale: this.scale / Math.abs(this.points.getMax('z') - this.points.getMin('z')),
        });
        for (const key of Object.keys(this.markers)) {
            const points = this.points.getNamed(key);
            if (points.length) {
                this[key] = new Points(points);
            }
        }
    }

    getFeatures() {
        if (this.features === undefined) {
            this.features = new FaceFeatures();
            for (const [key, isFeature] of Object.entries(this.markers)) {
                if (isFeature && this[key]) {
                    this.features.add(key, this[key]);
                }
            }
        }
        return this.features;
    }

    compare(face) {
        if (face) {
            return this.getFeatures().distance(face.getFeatures());
        }
    }
}

/**
 * Face landmarks features.
 *
 * @author Toha <tohenk@yahoo.com>
 */
class FaceFeatures {

    /**
     * Add landmark feature.
     *
     * @param {string} key Landmark key
     * @param {Points} points Landmark values
     */
    add(key, points) {
        this[key] = points instanceof Points ? points.flatten() : points;
    }

    /**
     * Calculate distance from referenced features.
     *
     * @param {FaceFeatures|object} features Referenced features
     * @returns {number}
     */
    distance(features) {
        let res;
        if (features) {
            if (Object.keys(this).length !== Object.keys(features).length) {
                throw new Error('Unable to compare between different face features!');
            }
            const values = {};
            for (const feature of Object.keys(this)) {
                const pairs = this[feature]
                    .map((v, k) => Point.from({x: v, y: features[feature][k]}));
                values[feature] = Math.sqrt(pairs
                    .map(p => p.x - p.y)
                    .reduce((a, b) => a + (b * b), 0));
            }
            res = Object.values(values)
                .reduce((a, b) => a + b, 0) / Object.values(values).length;
        }
        return res;
    }

    find(featuresList, confidence = 0.075) {
        let index, conf;
        for (const [idx, features] of Object.entries(featuresList)) {
            const dist = this.distance(features);
            if (dist <= confidence && (conf === undefined || dist < conf)) {
                conf = dist;
                index = idx;
            }
        }
        return [index, conf];
    }

    static from(data) {
        const feat = new this();
        for (const [k, v] of Object.entries(data)) {
            feat.add(k, v);
        }
        return feat;
    }
}

class Points {

    points = []

    constructor(points = null) {
        if (Array.isArray(points)) {
            this.points.push(...points.filter(a => a instanceof Point));
        }
    }

    getNamed(name) {
        return this.points.filter(p => p.name === name);
    }

    getCenter() {
        if (this.points.length) {
            if (this.center === undefined) {
                this.center = Point.from({
                    name: 'center',
                    x: this.getMean('x'),
                    y: this.getMean('y'),
                    z: this.getMean('z'),
                });
            }
            return this.center;
        }
    }

    getMean(axis) {
        return this.points
            .map(p => typeof p[axis] === 'function' ? p[axis]() : p[axis])
            .reduce((a, b) => a + b, 0) / this.points.length;
    }

    getMin(axis) {
        return Math.min(...this.points
            .map(p => p[axis]));
    }

    getMax(axis) {
        return Math.max(...this.points
            .map(p => p[axis]));
    }

    flatten() {
        const res = [];
        this.points.forEach(p => {
            res.push(...p.flatten());
        });
        return res;
    }

    normalize(args) {
        this.points.forEach(p => {
            p.normalize(args);
        });
        return this;
    }

    toJSON() {
        return this.points.map(p => p.toJSON());
    }

    static from(points) {
        const res = new this(points.map(p => Point.from(p)));
        return res;
    }
}

class Point {

    constructor({name, x, y, z}) {
        const s = (p, v) => {
            if (v !== undefined) {
                this[p] = v;
            }
        }
        for (const [k, v] of [
            ['name', name],
            ['x', x],
            ['y', y],
            ['z', z],
        ]) {
            s(k, v);
        }
    }

    flatten() {
        return [this.x, this.y, this.z];
    }

    normalize({xMin, yMin, zMin, xScale, yScale, zScale}) {
        const f = (k, v, op) => {
            if (v !== undefined && v !== null) {
                if (this.orig === undefined) {
                    this.orig = {};
                }
                this.orig[k] = this[k];
                switch (op) {
                    case '+':
                        this[k] += v;
                        break;
                    case '-':
                        this[k] -= v;
                        break;
                    case '*':
                        this[k] *= v;
                        break;
                    case '/':
                        this[k] /= v;
                        break;
                }
            }
        }
        for (const [k, v, op] of [
            ['x', xMin, '-'],
            ['y', yMin, '-'],
            ['z', zMin, '-'],
            ['x', xScale, '*'],
            ['y', yScale, '*'],
            ['z', zScale, '*'],
        ]) {
            f(k, v, op);
        }
        return this;
    }

    toJSON() {
        const res = {};
        for (const k of ['x', 'y', 'z', 'name']) {
            if (this[k] !== undefined) {
                res[k] = this[k];
            }
        }
        return res;
    }

    static from({name, x, y, z}) {
        const res = new this({name, x, y, z});
        return res;
    }
}

module.exports = {
    FaceDetection,
    FaceLandmark,
    FaceFeatures,
}