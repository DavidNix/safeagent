default: help

BASE_BRANCH ?= main
FEAT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD)

.PHONY: default help
help: ## Print this help message
	@echo "Available make commands:"; grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: check-clean
check-clean: ## Check if git state is clean
	@test -z "$$(git status --porcelain)" || { echo "Error: Git is dirty."; git status --short; exit 1; }

.PHONY: pr
pr: check-clean ## Mimic a local PR from a branch into upstream
	@if [ "$(FEAT_BRANCH)" = "$(BASE_BRANCH)" ]; then \
		echo "Error: You are on the $(BASE_BRANCH) branch."; \
		exit 1; \
	fi
	@git checkout $(BASE_BRANCH)
	@git merge --squash $(FEAT_BRANCH)

.PHONY: vet
vet: ## Run all linters and static checks
	$(MAKE) -C go vet

.PHONY: test
test: ## Run all tests
	$(MAKE) -C go test
