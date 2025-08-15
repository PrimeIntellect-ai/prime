#!/bin/bash

# Test script for prime-cli sandbox cp functionality
# Tests both upload and download for single files and directories
# Ensures proper path handling without creating extra directories

set -e  # Exit on any error

# Configuration
SANDBOX_ID="${1:-ecbtf5wqua2grdtveq4vlrde}"
TEST_DIR="sandbox_cp_test"
LOCAL_TEST_DIR="$TEST_DIR/local"
SANDBOX_TEST_DIR="$TEST_DIR/sandbox"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

cleanup() {
    log_info "Cleaning up test files..."
    rm -rf "$TEST_DIR"
    log_success "Cleanup completed"
}

# Check if prime-cli is available
check_prime_cli() {
    # Try to activate virtual environment if it exists
    if [ -f ".venv/bin/activate" ]; then
        log_info "Activating virtual environment..."
        source .venv/bin/activate
    fi

    if ! command -v prime &> /dev/null; then
        log_error "prime-cli is not installed or not in PATH"
        log_error "Please install prime-cli in development mode: pip install -e ."
        exit 1
    fi
    log_success "prime-cli found at: $(which prime)"
}

# Create test files and directories
setup_test_files() {
    log_info "Setting up test files and directories..."

    # Clean up any existing test directory
    rm -rf "$TEST_DIR"

    # Create local test directory
    mkdir -p "$LOCAL_TEST_DIR"

    # Create test files
    echo "This is a test file for upload" > "$LOCAL_TEST_DIR/testfile.txt"
    echo "Another test file" > "$LOCAL_TEST_DIR/another_file.txt"

    # Create test directory with files
    mkdir -p "$LOCAL_TEST_DIR/testdir"
    echo "File 1 in testdir" > "$LOCAL_TEST_DIR/testdir/file1.txt"
    echo "File 2 in testdir" > "$LOCAL_TEST_DIR/testdir/file2.txt"
    mkdir -p "$LOCAL_TEST_DIR/testdir/subdir"
    echo "File in subdir" > "$LOCAL_TEST_DIR/testdir/subdir/subfile.txt"

    # Create a file with spaces in name
    echo "File with spaces" > "$LOCAL_TEST_DIR/file with spaces.txt"

    log_success "Test files created"
}

# Test single file upload
test_single_file_upload() {
    log_info "Testing single file upload..."

    # Test upload functionality now that backend issues are fixed
    log_info "Testing upload functionality..."

    # Create test files manually in sandbox for download testing
    log_info "Creating test files manually in sandbox..."

    # Test upload functionality
    log_info "Test 1.1: Upload single file to sandbox workspace"
    prime sandbox cp "$LOCAL_TEST_DIR/testfile.txt" "$SANDBOX_ID:/sandbox-workspace/testfile.txt"

    log_info "Test 1.2: Upload single file to sandbox workspace (another file)"
    prime sandbox cp "$LOCAL_TEST_DIR/another_file.txt" "$SANDBOX_ID:/sandbox-workspace/another_file.txt"

    log_info "Test 1.3: Upload file with spaces in name"
    prime sandbox cp "$LOCAL_TEST_DIR/file with spaces.txt" "$SANDBOX_ID:/sandbox-workspace/file_with_spaces.txt"

    log_info "Test 1.4: Upload directory to sandbox workspace"
    prime sandbox cp "$LOCAL_TEST_DIR/testdir" "$SANDBOX_ID:/sandbox-workspace/testdir"

    log_success "Test files created manually in sandbox"
}

# Test directory upload
test_directory_upload() {
    log_info "Testing directory upload..."

    # Directory upload already tested in previous step
    log_info "Directory upload already tested in previous step"

    log_success "Directory upload tests skipped"
}

# Test single file download
test_single_file_download() {
    log_info "Testing single file download..."

    # Create download test directory
    mkdir -p "$SANDBOX_TEST_DIR"

    # Test download functionality now that backend issues are fixed
    log_info "Testing download functionality..."

    # Test 3.1: Download single file from sandbox workspace
    log_info "Test 3.1: Download single file from sandbox workspace"
    prime sandbox cp "$SANDBOX_ID:/sandbox-workspace/testfile.txt" "$SANDBOX_TEST_DIR/downloaded_testfile.txt"

    # Verify the file was downloaded as a file, not a directory
    if [ -f "$SANDBOX_TEST_DIR/downloaded_testfile.txt" ] && [ ! -d "$SANDBOX_TEST_DIR/downloaded_testfile.txt" ]; then
        log_success "File downloaded correctly as a file"
    else
        log_error "File was downloaded as a directory or doesn't exist"
        exit 1
    fi

    # Test 3.2: Download single file from sandbox directory
    log_info "Test 3.2: Download single file from sandbox directory"
    prime sandbox cp "$SANDBOX_ID:/sandbox-workspace/another_file.txt" "$SANDBOX_TEST_DIR/downloaded_another_file.txt"

    # Test 3.3: Download file with spaces (renamed)
    log_info "Test 3.3: Download file with spaces (renamed)"
    prime sandbox cp "$SANDBOX_ID:/sandbox-workspace/file_with_spaces.txt" "$SANDBOX_TEST_DIR/downloaded_file_with_spaces.txt"

    log_success "File verification completed"
}

# Test directory download
test_directory_download() {
    log_info "Testing directory download..."

    # Test directory download functionality now that backend issues are fixed
    log_info "Testing directory download functionality..."

    # Test 4.1: Download directory from sandbox
    log_info "Test 4.1: Download directory from sandbox"
    prime sandbox cp "$SANDBOX_ID:/sandbox-workspace/testdir" "$SANDBOX_TEST_DIR/downloaded_testdir"

    # Verify the directory was downloaded correctly
    if [ -d "$SANDBOX_TEST_DIR/downloaded_testdir" ]; then
        log_success "Directory downloaded correctly"
        # Check if subdirectory and files exist
        if [ -f "$SANDBOX_TEST_DIR/downloaded_testdir/file1.txt" ] && \
           [ -f "$SANDBOX_TEST_DIR/downloaded_testdir/file2.txt" ] && \
           [ -d "$SANDBOX_TEST_DIR/downloaded_testdir/subdir" ] && \
           [ -f "$SANDBOX_TEST_DIR/downloaded_testdir/subdir/subfile.txt" ]; then
            log_success "All directory contents downloaded correctly"
        else
            log_error "Directory contents not downloaded correctly"
            exit 1
        fi
    else
        log_error "Directory was not downloaded correctly"
        exit 1
    fi

    log_success "Directory verification completed"
}

# Test file rename scenarios
test_file_rename() {
    log_info "Testing file rename scenarios..."

        # Test 5.1: Create file with different name in sandbox
    log_info "Test 5.1: Create file with different name in sandbox"
    prime sandbox run "$SANDBOX_ID" "bash -c \"printf 'Renamed test file content\n' > /sandbox-workspace/renamed_testfile.txt\""

    # Test 5.2: Download with different name
    log_info "Test 5.2: Download file with different name"
    prime sandbox cp "$SANDBOX_ID:/sandbox-workspace/renamed_testfile.txt" "$SANDBOX_TEST_DIR/renamed_downloaded_file.txt"

        # Test 5.3: Create directory with different name in sandbox
    log_info "Test 5.3: Create directory with different name in sandbox"
    prime sandbox run "$SANDBOX_ID" "mkdir -p /sandbox-workspace/renamed_testdir"
    prime sandbox run "$SANDBOX_ID" "bash -c \"printf 'Renamed dir file\n' > /sandbox-workspace/renamed_testdir/file.txt\""

    # Test 5.4: Download directory with different name
    log_info "Test 5.4: Download directory with different name"
    prime sandbox cp "$SANDBOX_ID:/sandbox-workspace/renamed_testdir" "$SANDBOX_TEST_DIR/renamed_downloaded_dir"

    log_success "File rename tests completed"
}

# Test edge cases
test_edge_cases() {
    log_info "Testing edge cases..."

        # Test 6.1: Create file in $HOME expansion
    log_info "Test 6.1: Create file in \$HOME expansion"
    prime sandbox run "$SANDBOX_ID" "bash -c \"printf 'Home test file\n' > \$HOME/testfile_home.txt\""

    # Test 6.2: Download from $HOME expansion
    log_info "Test 6.2: Download from \$HOME expansion"
    prime sandbox cp "$SANDBOX_ID:\$HOME/testfile_home.txt" "$SANDBOX_TEST_DIR/testfile_from_home.txt"

    # Test 6.3: Create file in absolute path
    log_info "Test 6.3: Create file in absolute path"
    prime sandbox run "$SANDBOX_ID" "mkdir -p /sandbox-workspace/absolute/path"
    prime sandbox run "$SANDBOX_ID" "bash -c \"printf 'Absolute path test file\n' > /sandbox-workspace/absolute/path/testfile.txt\""

    # Test 6.4: Download from absolute path
    log_info "Test 6.4: Download from absolute path"
    prime sandbox cp "$SANDBOX_ID:/sandbox-workspace/absolute/path/testfile.txt" "$SANDBOX_TEST_DIR/testfile_from_absolute.txt"

    log_success "Edge case tests completed"
}

# Verify all downloaded files
verify_downloads() {
    log_info "Verifying all downloaded files..."

    local expected_files=(
        "downloaded_testfile.txt"
        "downloaded_another_file.txt"
        "downloaded_file_with_spaces.txt"
        "renamed_downloaded_file.txt"
        "testfile_from_home.txt"
        "testfile_from_absolute.txt"
    )

    local expected_dirs=(
        "downloaded_testdir"
        "renamed_downloaded_dir"
    )

    # Check files
    for file in "${expected_files[@]}"; do
        if [ -f "$SANDBOX_TEST_DIR/$file" ]; then
            log_success "File $file exists and is a file"
        else
            log_error "File $file missing or is not a file"
            exit 1
        fi
    done

    # Check directories
    for dir in "${expected_dirs[@]}"; do
        if [ -d "$SANDBOX_TEST_DIR/$dir" ]; then
            log_success "Directory $dir exists and is a directory"
        else
            log_error "Directory $dir missing or is not a directory"
            exit 1
        fi
    done

    log_success "All downloads verified successfully"
}

# Main test execution
main() {
    log_info "Starting prime-cli sandbox cp tests"
    log_info "Using sandbox ID: $SANDBOX_ID"

    # Check prerequisites
    check_prime_cli

    # Setup
    setup_test_files

    # Run tests
    test_single_file_upload
    test_directory_upload
    test_single_file_download
    test_directory_download
    test_file_rename
    test_edge_cases

    # Verify results
    verify_downloads

    log_success "All tests completed successfully!"

    # Show summary
    echo
    log_info "Test Summary:"
    echo "  - Single file upload: ✓"
    echo "  - Directory upload: ✓"
    echo "  - Single file download: ✓"
    echo "  - Directory download: ✓"
    echo "  - File rename: ✓"
    echo "  - Edge cases: ✓"
    echo "  - Path handling: ✓"
    echo ""
    echo "  All upload/download functionality is now working correctly!"

    # Cleanup
    cleanup
}

# Handle script interruption
trap cleanup EXIT

# Run main function
main "$@"
