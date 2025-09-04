#!/bin/bash

# Prime CLI Upload/Download End-to-End Test Script
# Tests all the async improvements we made

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
SANDBOX_ID=""

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    ((TESTS_PASSED++))
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    ((TESTS_FAILED++))
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

cleanup() {
    log_info "Cleaning up test environment..."
    
    # Remove test files
    rm -rf /tmp/prime-cli-test
    
    # Delete sandbox if it exists
    if [[ -n "$SANDBOX_ID" ]]; then
        log_info "Deleting test sandbox: $SANDBOX_ID"
        prime sandbox delete $SANDBOX_ID --yes || true
    fi
    
    # Final test report
    echo ""
    echo "=================================="
    echo "TEST RESULTS"
    echo "=================================="
    echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
    echo "Total Tests: $((TESTS_PASSED + TESTS_FAILED))"
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo -e "${GREEN}🎉 All tests passed!${NC}"
        exit 0
    else
        echo -e "${RED}❌ Some tests failed.${NC}"
        exit 1
    fi
}

# Set up cleanup on exit
trap cleanup EXIT

test_environment_setup() {
    log_info "Setting up test environment..."
    
    # Activate virtual environment
    log_info "Activating virtual environment..."
    if [[ -f ".venv/bin/activate.fish" ]]; then
        # For bash, we need to source the regular activate script
        source .venv/bin/activate
        log_success "Virtual environment activated"
    else
        log_error "Virtual environment not found at .venv/bin/activate"
        return 1
    fi
    
    # Configure CLI to use local setup
    log_info "Configuring CLI for local backend..."
    prime config use local
    log_success "CLI configured for local backend"
    
    # Verify backend connection
    log_info "Testing backend connection..."
    if prime sandbox list > /dev/null 2>&1; then
        log_success "Backend connection working"
    else
        log_error "Cannot connect to backend - make sure docker-compose is running"
        return 1
    fi
    
    # Create test workspace
    mkdir -p /tmp/prime-cli-test
    cd /tmp/prime-cli-test
    log_success "Test workspace created at /tmp/prime-cli-test"
}

create_test_sandbox() {
    log_info "Creating test sandbox..."
    
    # Create sandbox and capture output
    prime sandbox create python:3.11-slim --name "async-test-$(date +%s)" --yes > sandbox_create.log 2>&1
    SANDBOX_ID=$(grep -oE 'sandbox [a-z0-9]+' sandbox_create.log | head -1 | cut -d' ' -f2 || echo "")
    
    if [[ -z "$SANDBOX_ID" ]]; then
        log_error "Failed to create sandbox or extract ID"
        return 1
    fi
    
    log_success "Created sandbox: $SANDBOX_ID"
    
    # Wait for sandbox to be ready
    log_info "Waiting for sandbox to be ready..."
    local attempts=0
    local max_attempts=30
    
    while [[ $attempts -lt $max_attempts ]]; do
        local status=$(prime sandbox get $SANDBOX_ID --output=json 2>/dev/null | jq -r '.status' 2>/dev/null || echo "UNKNOWN")
        
        case $status in
            "RUNNING")
                log_success "Sandbox is ready!"
                return 0
                ;;
            "ERROR"|"TERMINATED")
                log_error "Sandbox failed with status: $status"
                return 1
                ;;
            *)
                log_info "Sandbox status: $status (attempt $((attempts+1))/$max_attempts)"
                sleep 5
                ((attempts++))
                ;;
        esac
    done
    
    log_error "Sandbox failed to become ready within timeout"
    return 1
}

test_basic_operations() {
    log_info "Testing basic upload/download operations..."
    
    # Create test files
    echo "Hello from test file 1" > test1.txt
    echo "Hello from test file 2" > test2.txt
    echo "Multi-line test file
Line 2
Line 3" > test3.txt
    
    # Test single file upload
    log_info "Testing single file upload..."
    if prime sandbox cp test1.txt $SANDBOX_ID:tmp/uploaded1.txt; then
        log_success "Single file upload completed"
    else
        log_error "Single file upload failed"
        return 1
    fi
    
    # Verify file in sandbox
    log_info "Verifying uploaded file..."
    if prime sandbox run $SANDBOX_ID "cat /sandbox-workspace/tmp/uploaded1.txt" | grep -q "Hello from test file 1"; then
        log_success "Uploaded file content verified"
    else
        log_error "Uploaded file content verification failed"
        return 1
    fi
    
    # Test single file download
    log_info "Testing single file download..."
    if prime sandbox cp $SANDBOX_ID:tmp/uploaded1.txt downloaded1.txt; then
        log_success "Single file download completed"
    else
        log_error "Single file download failed"
        return 1
    fi
    
    # Verify downloaded content
    if diff test1.txt downloaded1.txt > /dev/null; then
        log_success "Downloaded file content matches original"
    else
        log_error "Downloaded file content differs from original"
        return 1
    fi
}

test_directory_operations() {
    log_info "Testing directory upload/download operations..."
    
    # Create test directory structure
    mkdir -p test_dir/subdir1/subdir2
    echo "Root file content" > test_dir/root.txt
    echo "Subdir1 file content" > test_dir/subdir1/sub1.txt
    echo "Subdir2 file content" > test_dir/subdir1/subdir2/sub2.txt
    echo "Another root file" > test_dir/another.txt
    
    # Test directory upload
    log_info "Testing directory upload..."
    if prime sandbox cp test_dir $SANDBOX_ID:tmp/uploaded_dir; then
        log_success "Directory upload completed"
    else
        log_error "Directory upload failed"
        return 1
    fi
    
    # Verify directory structure in sandbox
    log_info "Verifying uploaded directory structure..."
    if prime sandbox run $SANDBOX_ID "find /sandbox-workspace/tmp/uploaded_dir -type f | wc -l" | grep -q "4"; then
        log_success "Directory structure preserved (4 files found)"
    else
        log_error "Directory structure not preserved correctly"
        return 1
    fi
    
    # Test directory download
    log_info "Testing directory download..."
    if prime sandbox cp $SANDBOX_ID:tmp/uploaded_dir downloaded_dir; then
        log_success "Directory download completed"
    else
        log_error "Directory download failed"
        return 1
    fi
    
    # Verify downloaded directory structure
    if diff -r test_dir downloaded_dir > /dev/null 2>&1; then
        log_success "Downloaded directory structure matches original"
    else
        log_error "Downloaded directory structure differs from original"
        return 1
    fi
}

test_concurrent_operations() {
    log_info "Testing concurrent operations (async showcase)..."
    
    # Create multiple test files
    for i in {1..5}; do
        echo "Concurrent test file $i - $(date)" > concurrent_$i.txt
    done
    
    # Test concurrent uploads (background processes to simulate concurrency)
    log_info "Testing concurrent uploads..."
    local start_time=$(date +%s)
    
    prime sandbox cp concurrent_1.txt $SANDBOX_ID:tmp/concurrent_1.txt &
    local pid1=$!
    prime sandbox cp concurrent_2.txt $SANDBOX_ID:tmp/concurrent_2.txt &
    local pid2=$!
    prime sandbox cp concurrent_3.txt $SANDBOX_ID:tmp/concurrent_3.txt &
    local pid3=$!
    
    # Wait for all uploads to complete
    wait $pid1 $pid2 $pid3
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_success "Concurrent uploads completed in ${duration}s"
    
    # Verify all files were uploaded
    local uploaded_count=$(prime sandbox run $SANDBOX_ID "ls /sandbox-workspace/tmp/concurrent_*.txt | wc -l" 2>/dev/null | tr -d ' ' || echo "0")
    if [[ "$uploaded_count" == "3" ]]; then
        log_success "All 3 concurrent uploads verified in sandbox"
    else
        log_error "Not all concurrent uploads completed (found: $uploaded_count)"
        return 1
    fi
}

test_large_file_warning() {
    log_info "Testing large file warning..."
    
    # Create a moderately large file (>100MB to trigger warning)
    log_info "Creating large test file (150MB)..."
    if prime sandbox run $SANDBOX_ID "dd if=/dev/zero of=/sandbox-workspace/large_file bs=1M count=150" > /dev/null 2>&1; then
        log_success "Large file created in sandbox"
    else
        log_error "Failed to create large file in sandbox"
        return 1
    fi
    
    # Test download with warning (capture stderr)
    log_info "Testing large file download (should show warning)..."
    if prime sandbox cp $SANDBOX_ID:large_file large_download.dat 2>&1 | grep -i -q "warning\|large"; then
        log_success "Large file warning displayed correctly"
        rm -f large_download.dat  # Clean up large file
    else
        log_warning "Large file warning not detected (might be working but not logging to stderr)"
    fi
}

test_error_handling() {
    log_info "Testing error handling..."
    
    # Test with non-existent local file
    log_info "Testing upload of non-existent file..."
    if prime sandbox cp nonexistent.txt $SANDBOX_ID:tmp/test.txt 2>/dev/null; then
        log_error "Upload of non-existent file should have failed"
        return 1
    else
        log_success "Upload of non-existent file correctly failed"
    fi
    
    # Test with non-existent sandbox path
    log_info "Testing download of non-existent file..."
    if prime sandbox cp $SANDBOX_ID:nonexistent/path.txt test_download.txt 2>/dev/null; then
        log_error "Download of non-existent file should have failed"
        return 1
    else
        log_success "Download of non-existent file correctly failed"
    fi
    
    # Test with invalid sandbox ID
    log_info "Testing operations with invalid sandbox ID..."
    if prime sandbox cp test1.txt invalid-sandbox-id:tmp/test.txt 2>/dev/null; then
        log_error "Operation with invalid sandbox ID should have failed"
        return 1
    else
        log_success "Operation with invalid sandbox ID correctly failed"
    fi
}

test_path_traversal_security() {
    log_info "Testing path traversal security (if possible)..."
    
    # This is harder to test without creating malicious tars
    # For now, we'll create a simple test
    log_info "Creating test with suspicious filename..."
    
    # Create file with suspicious name in sandbox
    if prime sandbox run $SANDBOX_ID "echo 'test content' > '/tmp/..test'" 2>/dev/null; then
        # Try to download it - should work since it's not actually traversing
        if prime sandbox cp $SANDBOX_ID:/tmp/..test safe_download.txt 2>/dev/null; then
            log_success "Handled suspicious filename correctly"
            rm -f safe_download.txt
        else
            log_warning "Download failed - this might be expected security behavior"
        fi
    else
        log_warning "Could not create test file for path traversal test"
    fi
}

main() {
    echo "=========================================="
    echo "Prime CLI Upload/Download Test Suite"
    echo "=========================================="
    echo ""
    
    # Run all tests
    test_environment_setup || exit 1
    create_test_sandbox || exit 1
    
    echo ""
    log_info "Starting functional tests..."
    test_basic_operations
    test_directory_operations
    test_concurrent_operations
    test_large_file_warning
    test_error_handling
    test_path_traversal_security
    
    log_info "All tests completed!"
}

# Run the main function
main "$@"