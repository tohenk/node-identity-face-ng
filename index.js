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
const debug = require('debug')('identity:face-api');

class FaceApiId extends Identity {

    VERSION = 'FACEIDENTITY-1.0'

    init() {
        super.init();
        this.id = 'FACEAPI';
        this.proxyServerId = 'FACEIDENTITY';
        this.channelType = 'cluster';
        this.workerOptions = {
            worker: path.join(__dirname, 'worker'),
            maxWorker: 4,
            maxWorks: 50,
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
                    return {face: await this.detectFaces(this.normalize(data.feature))};
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

    async getFace(img) {
        if (this.detector === undefined) {
            this.detector = new FaceDetection();
        }
        const landmark = await this.detector.getFace(img);
        if (landmark) {
            return new FaceLandmark(landmark);
        }
    }

    async getFaceFeatures(img) {
        const face = await this.getFace(img);
        if (face) {
            return face.getFeatures();
        }
    }

    async detectFaces(img) {
        const face = await this.getFace(img);
        if (face) {
            const box = {};
            for (const k of [['left', 'xMin'], ['top', 'yMin'], 'width', 'height']) {
                if (Array.isArray(k)) {
                    box[k[0]] = parseInt(face.box[k[1]]);
                } else {
                    box[k] = parseInt(face.box[k]);
                }
            }
            let res = sharp(img);
            // only crop when detected box is smalled then the image
            if ((box.left + box.width) < face.shape[1] && (box.top + box.height) < face.shape[0]) {
                res = res.extract(box);
            }
            return await res.toBuffer();
        }
    }

    async faceIdentify(feature, workid) {
        const feat = await this.getFaceFeatures(feature);
        if (feat) {
            return this.getIdentifier().identify(this.fixWorkId(workid), feat);
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

module.exports = FaceApiId;