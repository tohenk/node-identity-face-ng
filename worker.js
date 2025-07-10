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

const { Worker } = require('@ntlab/identity');
const { FaceDetection, FaceLandmark, FaceFeatures } = require('./face');
const debug = require('debug')('identity:worker:face-ng');

/**
 * @type {FaceDetection}
 */
let detector;
let stopped = false;

async function verify(work, start, end) {
    log('FACE> [%d] Verifying %s from %d to %d', Worker.id, work.id, start, end);
    const features = [];
    const indices = [];
    let count = 0;
    let matched = null;
    let current = start;
    try {
        // prepare trained data
        log('FACE> [%d] Preparing data...', Worker.id);
        while (current <= end) {
            if (stopped) {
                break;
            }
            const feature = await getFaceFeatures(work.items, current);
            if (feature) {
                features.push(feature);
                indices.push(current);
                count++;
            }
            current++;
        }
        if (!stopped && features.length) {
            // find best matches
            log('FACE> [%d] Find match...', Worker.id);
            const feature = FaceFeatures.from(work.feature);
            const [match, confidence] = feature.find(features);
            if (match !== undefined) {
                matched = {label: indices[match], confidence: 1 - confidence};
            }
        }
        // done
        log('FACE> [%d] Done verifying %d sample(s)', Worker.id, count);
    }
    catch (err) {
        error('FACE> [%d] Err: %s', Worker.id, err);
    }
    Worker.send({cmd: 'done', work: work, matched: matched, worker: Worker.id});
}

async function getFaces(img) {
    if (detector === undefined) {
        detector = new FaceDetection();
    }
    const detection = await detector.getFaces(img);
    if (detection.faces) {
        return detection.faces
            .map(landmark => new FaceLandmark({shape: detection.shape, ...landmark}));
    }
}

async function getFaceFeatures(items, index) {
    let res;
    const data = items[index];
    if (data) {
        if (data.type === 'Buffer' && data.data) {
            const buff = Buffer.from(data.data);
            const faces = await getFaces(buff);
            if (Array.isArray(faces) && faces.length) {
                res = faces[0].getFeatures();
                items[index] = res;
            } else {
                items[index] = null;
            }
            Worker.send({cmd: 'update', index, data: items[index], worker: Worker.id});
        } else {
            res = data;
        }
    }
    return res;
}

function log(...args) {
    debug(...args);
}

function error(...args) {
    debug(...args);
}

Worker.on('message', async (data) => {
    switch (data.cmd) {
        case 'do':
            stopped = false;
            await verify(data.work, data.start, data.end);
            break;
        case 'stop':
            stopped = true;
            log('FACE> [%d] Stopping', Worker.id);
            break;
    }
});
