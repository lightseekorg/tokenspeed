# Plugin server arguments

Runtime plugins register typed launch options independently of HF model
configuration. `ServerArgs.extra_server_args` is a flat dictionary of JSON
scalars. Field ownership is internal to the registry: two plugins cannot
register the same field, and unknown fields or invalid values fail before
numerical policy resolution. JSON input and individual CLI flags use the same
validation; conflicting explicit values are rejected.

`register_extra_server_args(namespace, config_type, validate=...)` registers a
frozen dataclass of scalar types, homogeneous `Literal` choices and optional
`None`. Model-derived fields do not belong in this schema. A field tagged with
`metadata={"native_server_arg": True}` binds the matching native `ServerArgs`
value; it is excluded from plugin JSON, flags and public field ownership.
Individual flags use `--<namespace>-<field>`, while JSON has no namespace wrapper.
An optional validator checks native argument interactions without initializing
execution components. Registration participates in the loader's rollback.

The lightweight plugin hook is also called `register_extra_server_args()`.
Headless startup uses it before parsing only when every enabled runtime plugin
provides the hook. Each hook registers the plugin's complete argument policy
without depending on runtime or kernel execution registrations. Full kernel
and runtime registration then runs before reporting readiness, replacing the
recorded lightweight registrations. If any enabled plugin lacks the hook, or
lightweight registration fails, all plugins load eagerly in the ordinary order.
No package names or plugin-specific dependency lists belong in the host loader.
Typed schemas and numerical envelopes follow the same lifecycle, so argument
parsing does not require GPU execution imports.

Numerical-policy resolution works on provisional launch inputs. Once
`ServerArgs` initialization finishes, `get_extra_server_args` publishes one
immutable object per plugin and launch. Target, draft and their components
share that object. Changing its input values after publication fails explicitly;
construct a new `ServerArgs` for a different launch. The cache belongs to
`ServerArgs`, not a process-global registry, and is dropped when copying or
pickling the launch. Workers rebuild their own shared object from plain values.

A model profile can return those shared arguments from
`create_runtime_config(model_config, server_args)`. The result is held in
`ModelConfig.runtime_config` and passed explicitly to the model constructor.
An optional `create_model_kwargs(model_config, load_config)` supplies other
constructor inputs from the actual model and loading configurations. It cannot
replace keywords already supplied by the host. This gives checkpoint loaders
and workspace owners their own inputs without storing model state in shared
arguments or HF metadata. Models without these hooks keep the existing contract.

Models and components own resolved arithmetic choices, buffers, graph state and
weight-update lifecycles. Their initialization must not mutate shared arguments.
Numerical policy continues to constrain capability planning and execution.

Plugin execution options enter the cache producer identity. Runtime-config
models include an empty options marker even when no values were supplied,
isolating their new launch-owned policy from older checkpoint-owned policy.
This preserves cache correctness across the configuration migration without
changing namespaces for unaffected deployments.
