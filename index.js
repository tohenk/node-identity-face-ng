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

const path = require('path');
const { Identity } = require('@ntlab/identity');
const { FaceDetection, FaceLandmark } = require('./face');
const sharp = require('sharp');
const debug = require('debug')('identity:face-ng');

class FaceId extends Identity {

    VERSION = 'FACEIDENTITY-1.0'

    init() {
        super.init();
        this.id = 'FACE';
        this.proxyServerId = 'FACEIDENTITY';
        this.channelType = 'cluster';
        this.workerOptions = {
            worker: path.join(__dirname, 'worker'),
            maxWorks: 0,
            hasConfidence: true,
        }
    }

    getCommands() {
        return {
            [Identity.MODE_ALL]: {
                'self-test': data => this.VERSION,
                'connect': data => true,
            },
            [Identity.MODE_VERIFIER]: {
                'identify': async (data) => {
                    return await this.faceIdentify(this.normalize(data.feature), data.workid);
                },
                'detect': async (data) => {
                    return await this.detectFaces(this.normalize(data.feature), data.options);
                },
                'count-template': data => {
                    return {count: this.getIdentifier().count()};
                },
                'reg-template': data => {
                    if (data.id && data.template) {
                        if (data.force && this.getIdentifier().has(data.id)) {
                            this.getIdentifier().remove(data.id);
                        }
                        const success = this.getIdentifier().add(data.id, this.normalize(data.template));
                        debug(`Register template ${data.id} [${success ? 'OK' : 'FAIL'}]`);
                        if (success) {
                            return {id: data.id};
                        }
                    }
                },
                'unreg-template': data => {
                    if (data.id) {
                        const success = this.getIdentifier().remove(data.id);
                        debug(`Unregister template ${data.id} [${success ? 'OK' : 'FAIL'}]`);
                        if (success) {
                            return {id: data.id};
                        }
                    }
                },
                'has-template': data => {
                    if (data.id) {
                        const success = this.getIdentifier().has(data.id);
                        if (success) {
                            return {id: data.id};
                        }
                    }
                },
                'clear-template': data => {
                    this.getIdentifier().clear();
                    return true;
                }
            }
        }
    }

    normalize(data) {
        if (typeof data === 'string') {
            const buff = new Uint8Array(data.length);
            for (let i = 0; i < data.length; i++) {
                buff[i] = data.charCodeAt(i);
            }
            data = buff;
        }
        return data;
    }

    async getFaces(img) {
        if (this.detector === undefined) {
            this.detector = new FaceDetection();
        }
        const detection = await this.detector.getFaces(img);
        if (detection.faces) {
            return detection.faces
                .map(landmark => new FaceLandmark({shape: detection.shape, ...landmark}));
        }
    }

    async getFaceFeatures(img) {
        const faces = await this.getFaces(img);
        if (Array.isArray(faces) && faces.length) {
            return faces.
                map(face => face.getFeatures());
        }
    }

    async detectFaces(img, options = null) {
        options = options || {};
        if (options.face === undefined) {
            options.face = true;
        }
        if (options.feature === undefined) {
            options.feature = true;
        }
        const res = [];
        const faces = await this.getFaces(img);
        if (Array.isArray(faces) && faces.length) {
            for (const face of faces) {
                const data = {};
                if (options.face) {
                    const box = {};
                    for (const k of [['left', 'xMin'], ['top', 'yMin'], 'width', 'height']) {
                        if (Array.isArray(k)) {
                            box[k[0]] = parseInt(face.box[k[1]]);
                        } else {
                            box[k] = parseInt(face.box[k]);
                        }
                    }
                    const faceimg = sharp(img);
                    // only crop when detected box is smaller then the image
                    if ((box.left + box.width) < face.shape[1] && (box.top + box.height) < face.shape[0]) {
                        faceimg.extract(box);
                        if (this.options.size) {
                            const scale = this.options.size / Math.max(box.width, box.height);
                            faceimg.resize(Math.ceil(box.width * scale), Math.ceil(box.height * scale));
                        }
                    }
                    data.face = await faceimg.toBuffer();
                }
                if (options.feature) {
                    data.features = face.getFeatures();
                }
                res.push(data);
            }
        }
        return res;
    }

    async faceIdentify(feature, workid) {
        const features = await this.getFaceFeatures(feature);
        if (Array.isArray(features)) {
            return this.getIdentifier().identify(this.fixWorkId(workid), features[0]);
        }
    }

    fixWorkId(workid) {
        if (!workid) {
            workid = Identity.genId();
        }
        return workid;
    }

    onreset() {
        this.doCmd(this.getPrefix('clear-template'));
    }
}

module.exports = FaceId;