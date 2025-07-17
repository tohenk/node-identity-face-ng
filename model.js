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
const path = require('path');
const { io } = require('@tensorflow/tfjs-core');
const gunzip = require('gunzip-maybe');
const tar = require('tar-stream');

/**
 * Provides local model to be used in tfjs detection.
 *
 * Example:
 *
 * ```js
 * const LocalModel = require('@ntlab/identity-face-ng/model');
 *
 * async function getConfig(refineLandmarks) {
 *     return {
 *         runtime: 'tfjs',
 *         refineLandmarks,
 *         detectorModelUrl: await LocalModel.create('face-detection', 'short'),
 *         landmarkModelUrl: await LocalModel.create('face-landmarks-detection', refineLandmarks ? 'attention-mesh' : 'face-mesh'),
 *     }
 * }
 * ```
 * @author Toha <tohenk@yahoo.com>
 */
class TfjsLocalModel {

    /**
     * Constructor.
     *
     * @param {string} name Model name
     * @param {string} variation Model variation
     */
    constructor(name, variation) {
        this.name = name;
        this.variation = variation;
    }

    /**
     * Read content of readable stream.
     *
     * @param {ReadableStream} stream 
     * @returns {Promise<Buffer>}
     */
    readStream(stream) {
        return new Promise((resolve, reject) => {
            const buff = [];
            stream.on('data', data => {
                buff.push(data);
            });
            stream.on('error', err => reject(err));
            stream.on('end', () => {
                resolve(Buffer.concat(buff));
            });
            stream.resume();
        });
    }

    /**
     * Extract model artifacts from tar archive.
     *
     * @param {string} filename Tar archived model file name
     * @returns {Promise<object>}
     */
    async extractArtifacts(filename) {
        const res = {};
        const streams = {};
        const extract = tar.extract();
        fs.createReadStream(filename)
            .pipe(gunzip())
            .pipe(extract);
        for await (const entry of extract) {
            streams[entry.header.name] = await this.readStream(entry);
        }
        const modelJson = 'model.json';
        if (streams[modelJson] === undefined) {
            throw new Error(`${filename} is not a model!`);
        }
        res.modelJson = JSON.parse(streams[modelJson]);
        res.loadWeights = async (manifest) => {
            const specs = io.getWeightSpecs(manifest);
            const datas = [];
            for (const entry of manifest) {
                const buff = [];
                for (const filepath of entry.paths) {
                    if (!streams[filepath]) {
                        throw new Error(`File ${filepath} is not found!`);
                    }
                    buff.push(streams[filepath]);
                }
                datas.push(Buffer.concat(buff));
            }
            return [specs, datas];
        }
        return res;
    }

    /**
     * Get the model artifacts.
     *
     * @returns {Promise<io.ModelArtifacts>}
     */
    async findArtifacts() {
        if (this.artifacts === undefined) {
            const model = [this.name, this.variation].join('/');
            const models = this.constructor.models;
            if (models[model] === undefined) {
                throw new Error(`Model ${this.name} with variation ${this.variation} is not exist!`);
            }
            const filename = Object.keys(models[model]).pop();
            const {modelJson, loadWeights} = await this.extractArtifacts(filename);
            this.artifacts = await io.getModelArtifactsForJSON(modelJson, loadWeights);
        }
        return this.artifacts;
    }

    /**
     * Create model handler.
     *
     * @returns {Promise<io.IOHandlerSync>}
     */
    async factory() {
        return io.fromMemorySync(await this.findArtifacts());
    }

    /**
     * Get all model tar archive from model directory.
     *
     * @property {object}
     */
    static get models() {
        if (this._models === undefined) {
            if (this._modeldir === undefined) {
                this._modeldir = path.join(__dirname, 'model');
            }
            this._models = {};
            fs.readdirSync(this._modeldir, {withFileTypes: true})
                .filter(f => f.isFile() && f.name.endsWith('.tar.gz'))
                .forEach(f => {
                    const parts = f.name.substr(0, f.name.length - 7).split('-tfjs-');
                    if (parts.length > 1) {
                        const vers = parts[1].match(/\-v(\d+)/);
                        if (vers.length) {
                            const model = [parts[0], parts[1].substr(0, parts[1].length - vers[0].length)].join('/');
                            if (this.models[model] === undefined) {
                                this.models[model] = {};
                            }
                            this.models[model][path.join(this._modeldir, f.name)] = parseInt(vers[1]);
                            if (Object.keys(this.models[model]).length > 1) {
                                this.models[model].sort((a, b) => a - b);
                            }
                        }
                    }
                });
        }
        return this._models;
    }

    /**
     * Set directory to look for model tar archive.
     *
     * @param {string} dir Model directory
     */
    static setModelDir(dir) {
        this._modeldir = dir;
    }

    /**
     * Create local model.
     *
     * @param {string} name Model name
     * @param {string} variation Model variation
     * @returns {Promise<io.IOHandlerSync>}
     */
    static create(name, variation) {
        const model = new this(name, variation);
        return model.factory();
    }
}

module.exports = TfjsLocalModel;