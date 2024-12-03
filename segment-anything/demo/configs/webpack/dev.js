// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// development config
const { merge } = require("webpack-merge");
const commonConfig = require("./common");
const { resolve } = require("path");

module.exports = merge(commonConfig, {
  mode: "development",
  devServer: {
    hot: true, // enable HMR on the server
    open: true,
    static: {
      directory: resolve(__dirname, 'public'),
    },
    // These headers enable the cross origin isolation state
    // needed to enable use of SharedArrayBuffer for ONNX 
    // multithreading. 
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "credentialless",
    },
    client: {
      webSocketURL: 'ws://localhost:8081/ws',
    },
  },
  devtool: "cheap-module-source-map",
});
