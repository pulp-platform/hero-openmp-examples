# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/), and this project adheres to
[Semantic Versioning](http://semver.org).

## v1.2.0 - 2018-09-25

### Added
- Added top level 'Makefile' for CI and automatic test generation

### Changed
- On 'make.inc':
	- Removed useless 'pulp' related cflags
	- Enable overridable 'run' and 'clean' makefile targets

### Fixed
- Added 'omp.h' on all the applications after #22 bug fix.

## v1.1.0 - 2018-09-18

### Added
- Add sobel-filter application.

## v1.0.1 - 2018-09-17

### Fixed
- Add license header to all C files.
- Ensure that remote directory exists during `make install`.
- `common/bench.h:` Add missing include guard.

## v1.0.0 - 2018-09-14

Initial public release
